"""DINOv2-SALAD identity representation and CAD-source training."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ['XFORMERS_DISABLED']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,os.environ.get('DINOV2_REPO',str(ROOT/'facebookresearch-dinov2-7764ea0')))
from dinov2.hub.backbones import dinov2_vitl14
from pytorch_metric_learning import losses,miners
spec=importlib.util.spec_from_file_location('pinned_salad',HERE/'official_source/models/aggregators/salad.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
DEADLINE=float('inf')

def budget():
    if time.monotonic()>DEADLINE:raise RuntimeError('Registered preparation/compute deadline reached; keep incomplete state')

def dump(name,x):
    (HERE/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=dinov2_vitl14(pretrained=False)
        path=Path(os.environ['DINOV2_PRETRAINED'])
        self.backbone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        for v in self.backbone.parameters():v.requires_grad_(False)
        assert len(self.backbone.blocks)==24
        for block in self.backbone.blocks[-4:]:
            for v in block.parameters():v.requires_grad_(True)
        for v in self.backbone.norm.parameters():v.requires_grad_(True)
        self.aggregation=official.SALAD(num_channels=1024,num_clusters=64,cluster_dim=128,token_dim=256,dropout=.3)

    def features(self,x):
        with torch.no_grad():
            t=self.backbone.prepare_tokens_with_masks(x)
            for b in self.backbone.blocks[:-4]:t=b(t)
        t=t.detach()
        for b in self.backbone.blocks[-4:]:t=b(t)
        t=self.backbone.norm(t)
        assert t.shape[1:]==(257,1024)
        return t[:,1:].reshape(len(t),16,16,1024).permute(0,3,1,2),t[:,0]

    def forward(self,x):return self.aggregation(self.features(x))

def input_tensor(a,training):
    x=torch.as_tensor(a,device='cuda').float().permute(0,3,1,2)/255
    if training:
        bright=.9+.2*torch.rand(len(x),1,1,1,device=x.device)
        contrast=.9+.2*torch.rand(len(x),1,1,1,device=x.device)
        mean=x.mean((2,3),keepdim=True)
        x=((x-mean)*contrast+mean).mul(bright).clamp(0,1)
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std

def image(path):
    budget()
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).copy()

def macro(r,records):
    identities={}
    for rank,row in zip(r,records):identities.setdefault(row['model_id'],[]).append(float(rank))
    return {**{'top1' if k==1 else 'recall_at_'+str(k):float(np.mean([np.mean(np.asarray(v)<=k) for v in identities.values()])) for k in [1,5,10,20,100]},
            'mrr':float(np.mean([np.mean(1/np.asarray(v)) for v in identities.values()]))}

def rank(score,truth):
    ix=np.arange(score.shape[1]);gt=score[np.arange(len(truth)),truth]
    return (1+(score>gt[:,None]).sum(1)+((score==gt[:,None])&(ix[None]<truth[:,None])).sum(1)).astype(int)

def paired_ci(new,old,records,seed_value):
    # Identity macro contribution resampled at association-group level.
    sizes={}
    for row in records:sizes[row['model_id']]=sizes.get(row['model_id'],0)+1
    groups={}
    for i,row in enumerate(records):groups.setdefault(row['group_id'],[]).append(i)
    gc=list(groups);d=[];w=[]
    for g in gc:
        indices=groups[g];weights=np.asarray([1/sizes[records[i]['model_id']] for i in indices])
        w.append(weights.sum());d.append(float(((new[indices]-old[indices])*weights).sum()))
    rng=np.random.default_rng(seed_value);idx=rng.integers(len(gc),size=(10000,len(gc)))
    values=np.asarray(d)[idx].sum(1)/np.asarray(w)[idx].sum(1)
    return {'difference':float(sum(d)/sum(w)),'ci95':np.quantile(values,[.025,.975]).tolist()}

@torch.inference_mode()
def extract(model,arrays,cls=False):
    model.eval();out=[]
    for start in range(0,len(arrays),32):
        budget()
        x=input_tensor(arrays[start:start+32],False)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if cls:
                f=F.normalize(model.features(x)[1].float(),dim=1)
            else:f=model(x).float()
        assert torch.isfinite(f).all()
        out.append(f.cpu().numpy())
    return np.concatenate(out)

def score(q,templates):
    result=[]
    for start in range(0,len(q),24):
        budget()
        s=q[start:start+24]@templates.reshape(-1,templates.shape[-1]).T
        s=s.reshape(-1,len(templates),templates.shape[1])
        result.append(np.partition(s,-5,axis=-1)[:,:,-5:].mean(-1))
    return np.concatenate(result)

def batches(p,bank,photos,stage,epoch):
    rng=np.random.default_rng(p['seed']+10000*int(stage=='real')+epoch)
    n=len(bank);views=np.asarray(p['source_train_views'])
    if stage=='source':
        per_model={i:rng.permutation(views) for i in range(n)}
        for vstart in range(0,len(views),4):
            order=rng.permutation(n)
            for start in range(0,n,16):
                ids=order[start:start+16];aa=[];yy=[]
                for mid in ids:
                    v=per_model[int(mid)][vstart:vstart+4]
                    if len(v)<4:v=np.concatenate([v,rng.choice(views,4-len(v),replace=False)])
                    aa.extend(bank[mid,v]);yy.extend([int(mid)]*4)
                yield np.stack(aa),np.asarray(yy)
    else:
        by_id={};ix={v:i for i,v in enumerate(p['gallery_model_ids'])}
        for row in p['train_records']:by_id.setdefault(ix[row['model_id']],[]).append(row)
        order=rng.permutation(sorted(by_id))
        for start in range(0,len(order),8):
            ids=order[start:start+8];aa=[];yy=[]
            for mid in ids:
                rows=sorted(by_id[int(mid)],key=lambda r:r['query_id'])
                row=rows[epoch%len(rows)]
                a=photos[row['query_id']]
                aa.extend([a,a]);aa.extend(bank[mid,rng.choice(views,2,replace=False)])
                yy.extend([int(mid)]*4)
            other=rng.choice(np.setdiff1d(np.arange(n),ids),len(ids),replace=False)
            for mid in other:
                aa.extend(bank[mid,rng.choice(views,4,replace=False)]);yy.extend([int(mid)]*4)
            yield np.stack(aa),np.asarray(yy)

def checkpoint(model,name,p,extra=None):
    names={n for n,v in model.named_parameters() if v.requires_grad}
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if k in names}
    torch.save({'trainable_state':state,'protocol_sha256':sha(HERE/'protocol.json'),'base_weight_sha256':p['base_weight_sha256'],
                'extra':extra or {}},HERE/name)
    return sha(HERE/name)

def train_stage(model,p,bank,photos,stage,total_budget,start_all):
    model.train();params=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=6e-5,weight_decay=9.5e-9)
    loss_fn=losses.MultiSimilarityLoss(alpha=2,beta=50,base=.5)
    miner=miners.MultiSimilarityMiner(epsilon=.1)
    epochs=p['training'][stage+'_epochs']
    steps=math.ceil(len(bank)/16)*math.ceil(len(p['source_train_views'])/4) if stage=='source' else math.ceil(len({r['model_id'] for r in p['train_records']})/8)
    schedule=torch.optim.lr_scheduler.LinearLR(opt,start_factor=1,end_factor=.2,total_iters=epochs*steps)
    history=[];timings=[];global_step=0
    for epoch in range(epochs):
        model.train();ls=[];nonzero=0;tick_epoch=time.monotonic()
        for a,y in batches(p,bank,photos,stage,epoch):
            if time.monotonic()-start_all>=total_budget:
                checkpoint(model,'unfinished.pt',p,{'stage':stage,'epoch':epoch,'step':global_step,'history':history})
                raise RuntimeError('Registered total compute budget exhausted; incomplete, not method failure')
            tick=time.monotonic();opt.zero_grad(set_to_none=True)
            x=input_tensor(a,True);y=torch.as_tensor(y,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
            z=z.float();pairs=miner(z,y);loss=loss_fn(z,y,pairs)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            if global_step==0 and stage=='source':
                # Independent scalar form of published MS objective for exactly the official mined pairs.
                sim=F.normalize(z.detach(),dim=1)@F.normalize(z.detach(),dim=1).T
                pa,pi,na,ni=[t.detach() for t in pairs];scalar=[]
                for ai in range(len(z)):
                    ps=sim[ai,pi[pa==ai]];ns=sim[ai,ni[na==ai]]
                    pl=torch.logsumexp(torch.cat([ps.new_zeros(1),-2*(ps-.5)]),0)/2
                    nl=torch.logsumexp(torch.cat([ns.new_zeros(1),50*(ns-.5)]),0)/50
                    scalar.append(pl+nl)
                reference=torch.stack(scalar).mean();error=float((reference-loss.detach()).abs())
                assert error<1e-5, 'Official MS loss/reference mismatch'
                dump('loss_check.json',{'status':'passed','reference_max_absolute_error':error,'anchor_count':len(z),'positive_pairs':len(pa),'negative_pairs':len(na),'gradient_loss_used':'official library, not reference'})
            loss.backward()
            total_grad=float(sum(v.grad.detach().abs().sum().item() for v in params if v.grad is not None))
            if not math.isfinite(total_grad):raise FloatingPointError('Nonfinite gradient')
            if total_grad>0:nonzero+=1
            if global_step<8 and stage=='source':
                prefixes=[v for n,v in model.backbone.named_parameters() if not v.requires_grad]
                assert all(v.grad is None for v in prefixes)
                if global_step==0:
                    trained={name:sum(v.grad.detach().abs().sum().item() for v in part.parameters() if v.grad is not None) for name,part in [('last_block',model.backbone.blocks[-1]),('aggregation',model.aggregation)]}
                    assert all(t>0 for t in trained.values())
                    dump('implementation_check.json',{'status':'passed','effective_batch':len(a),'unit_descriptor_max_error':float((z.norm(dim=1)-1).abs().max()),
                         'gradients':trained,'frozen_prefix_gradients_absent':True,'loss_library':'pytorch-metric-learning2.9.0',
                         'inference_api':'score(photo_descriptors,CAD_template_descriptors); no truth input','dtype':'bf16 autocast,float32 loss'})
            opt.step();schedule.step();torch.cuda.synchronize()
            timings.append(time.monotonic()-tick);ls.append(float(loss.detach()));global_step+=1
            if global_step==8 and stage=='source':
                total_steps=p['training']['source_epochs']*steps+p['training']['real_epochs']*math.ceil(len({r['model_id'] for r in p['train_records']})/8)
                estimate=float(np.median(timings))*total_steps*1.3
                dump('timing_check.json',{'first8_batch_seconds':timings,'estimated_training_seconds_with30percent_margin':estimate,
                                          'budget_seconds':total_budget,'evaluation_reserve_seconds':1200,'gpu_name':torch.cuda.get_device_name(0),
                                          'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
                if estimate+1200>total_budget:
                    checkpoint(model,'unfinished.pt',p,{'reason':'cost estimate exceeds budget','steps':global_step})
                    raise RuntimeError('Fixed complete schedule cannot fit registered budget; no epoch shortening')
            if global_step%40==0:print(json.dumps({'stage':stage,'epoch':epoch+1,'step':global_step,'last_loss':ls[-1],'elapsed_seconds':time.monotonic()-start_all}),flush=True)
        item={'stage':stage,'epoch':epoch+1,'loss_mean':float(np.mean(ls)),'nonzero_gradient_batches':nonzero,'batches':len(ls),'epoch_seconds':time.monotonic()-tick_epoch}
        if nonzero==0:raise RuntimeError('Entire epoch had zero mined learning signal; configuration stopped')
        history.append(item);dump(stage+'_history.json',history);print(json.dumps(item),flush=True)
    ckpt_sha=checkpoint(model,stage+'_final.pt',p,{'history':history})
    return ckpt_sha

def evaluate(model,p,bank,photos,stage,cls=False):
    templates=extract(model,bank.reshape(-1,224,224,3),cls).reshape(len(bank),42,-1)
    rows=p['evaluation_records'];q=extract(model,np.stack([photos[r['query_id']] for r in rows]),cls)
    scores=score(q,templates)
    ix={v:i for i,v in enumerate(p['gallery_model_ids'])};truth=np.asarray([ix[r['model_id']] for r in rows]);rr=rank(scores,truth)
    np.savez_compressed(HERE/(stage+'_scores.npz'),scores=scores,query_ids=np.asarray([r['query_id'] for r in rows]),gallery_ids=np.asarray(p['gallery_model_ids']),ranks=rr)
    metrics=macro(rr,rows)
    source_records=[{'model_id':mid} for mid in p['gallery_model_ids'] for _ in p['source_validation_views']]
    source_q=templates[:,p['source_validation_views']].reshape(-1,templates.shape[-1]);source_s=score(source_q,templates[:,p['source_train_views']])
    source_truth=np.repeat(np.arange(len(bank)),len(p['source_validation_views']));source_r=rank(source_s,source_truth)
    np.savez_compressed(HERE/(stage+'_source_diagnostic.npz'),ranks=source_r,gallery_ids=np.asarray(p['gallery_model_ids']),query_views=np.asarray(p['source_validation_views']))
    result={'stage':stage,'photos':metrics,'source_held_view_diagnostic':macro(source_r,source_records),'source_reference_views':p['source_train_views'],'source_query_views':p['source_validation_views'],
            'queries_retained':len(rows),'gallery':len(bank),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    dump(stage+'_results.json',result);print(json.dumps(result),flush=True)
    del templates
    return rr,result

def main():
    global DEADLINE
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'));seed(p['seed']);torch.set_num_threads(8)
    status={'status':'preparing','planned_training_runs':1,'trained_stages':[],'start_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    begin=time.monotonic()
    DEADLINE=begin+p['budget']['preparation_wall_seconds']
    try:
        for source in p['source_snapshot']:assert sha(HERE/source['path'])==source['sha256']
        weight=Path(os.environ['DINOV2_PRETRAINED'])
        p['base_weight_sha256']=sha(weight)
        assert p['base_weight_sha256']=='d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'
        rows=p['train_records']+p['evaluation_records'];photos={r['query_id']:image(ROOT/r['repaired_path']) for r in rows}
        bank=np.empty((438,42,224,224,3),dtype=np.uint8)
        for i,mid in enumerate(p['gallery_model_ids']):
            for v in range(42):bank[i,v]=image(ROOT/'modules/cache/renders_sphere42'/mid/f'view_{v:02d}.png')
        status['input_images']=len(photos);status['templates']=438*42;status['RAM_image_bytes']=bank.nbytes
        if time.monotonic()-begin>p['budget']['preparation_wall_seconds']:raise RuntimeError('Preparation budget exceeded')
        model=Model().cuda();status['model_trainable_parameters']=sum(v.numel() for v in model.parameters() if v.requires_grad);dump('run_status.json',status)
        if args.preflight_only:
            status['status']='preflight_only_inputs_and_model_loaded';dump('run_status.json',status);return
        timed=time.monotonic();DEADLINE=timed+p['budget']['training_and_evaluation_wall_seconds'];status['status']='running';dump('run_status.json',status)
        # Frozen raw DINO under the SAME unchanged-image preprocessing is a controlled input baseline.
        frozen_r,frozen=evaluate(model,p,bank,photos,'frozen_DINO_same_input',cls=True)
        for stage in ['source','real']:
            ckpt=train_stage(model,p,bank,photos,stage,p['budget']['training_and_evaluation_wall_seconds'],timed)
            rr,res=evaluate(model,p,bank,photos,stage)
            status['trained_stages'].append({'stage':stage,'checkpoint_sha256':ckpt,'metrics':res['photos']})
            status['elapsed_seconds']=time.monotonic()-timed;dump('run_status.json',status)
        comparisons={}
        with (ROOT/'modules/score_cache/per_query.csv').open(encoding='utf-8-sig',newline='') as f:
            history=[r for r in csv.DictReader(f) if r['gallery']=='438']
        queryids=[r['query_id'] for r in p['evaluation_records']]
        comparisons['frozen_DINO_same_input']={'metrics':frozen['photos'],
                                             'top1_gain':paired_ci((rr==1).astype(float),(frozen_r==1).astype(float),p['evaluation_records'],p['seed']),
                                             'mrr_gain':paired_ci(1/rr.astype(float),1/frozen_r.astype(float),p['evaluation_records'],p['seed'])}
        for method in sorted({r['method'] for r in history}):
            table={r['query_id']:int(r['rank']) for r in history if r['method']==method}
            old=np.asarray([table[q] for q in queryids]);assert len(old)==97
            comparisons[method]={'metrics':macro(old,p['evaluation_records']),
                                 'top1_gain':paired_ci((rr==1).astype(float),(old==1).astype(float),p['evaluation_records'],p['seed']),
                                 'mrr_gain':paired_ci(1/rr.astype(float),1/old.astype(float),p['evaluation_records'],p['seed'])}
        dump('comparisons.json',comparisons)
        strongest=max(comparisons,key=lambda m:comparisons[m]['metrics']['mrr'])
        comp=comparisons[strongest];gain=((comp['top1_gain']['difference']>=.05 and comp['top1_gain']['ci95'][0]>0) or (comp['mrr_gain']['difference']>=.03 and comp['mrr_gain']['ci95'][0]>0))
        no_retreat=all(res['photos'][key]>=comp['metrics'][key]-.01 for key in ['recall_at_5','recall_at_20','recall_at_100'])
        status.update(status='complete_fixed_baseline',elapsed_seconds=time.monotonic()-timed,strongest_comparator=strongest,
                      development_gate_passed=bool(gain and no_retreat),original_method_established=False,
                      enhancement_training_started=False,new_independent_test=False)
        dump('run_status.json',status);print(json.dumps(status),flush=True)
    except Exception as e:
        status.update(status='incomplete_execution',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.monotonic()-begin)
        dump('run_status.json',status);print(json.dumps(status),flush=True);raise

if __name__=='__main__':main()
