"""One registered three-arm experiment; no model or parameter search."""
import os
os.environ['XFORMERS_DISABLED'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import gc, hashlib, importlib.util, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'modules/appearance'
spec = importlib.util.spec_from_file_location('fixed_parent', PARENT/'run_baseline.py')
parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
END = float('inf')
POLICY_ARM = None
POLICIES = None

def dump(name, data):
    (HERE/name).write_text(json.dumps(data, indent=2), encoding='utf-8')

def check():
    if time.monotonic() > END:
        raise RuntimeError('Registered combined budget reached; incomplete, not negative result')

def unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

def appearance(normal, mask, rng):
    color = rng.uniform(.35, .9, size=3)
    light = np.r_[rng.uniform(-.7, .7, size=2), 1.]
    light /= np.linalg.norm(light)
    return color, light, rng.uniform(.15, .4), rng.uniform(.45, .75)

def rgb(normal, mask, params):
    color, light, ambient, diffuse = params
    intensity = ambient + diffuse*np.maximum(np.asarray(normal, np.float32)@light, 0)
    a = np.uint8(np.clip(intensity[..., None]*color*255, 0, 255))*mask[..., None]
    return np.asarray(Image.fromarray(a).resize((224,224), Image.Resampling.BILINEAR)).copy()

def batches(p, bank, photos, geo, epoch):
    rng = np.random.default_rng(p['seed']+10000+epoch)
    ix = {m:i for i,m in enumerate(p['gallery_model_ids'])}
    by_id = {}
    for row in p['train_records']:
        by_id.setdefault(ix[row['model_id']], []).append(row)
    order = rng.permutation(sorted(by_id))
    views = p['source_train_views']
    for start in range(0, len(order), 8):
        check(); ids = order[start:start+8]; aa=[]; yy=[]
        for mid in ids:
            rows=sorted(by_id[int(mid)], key=lambda r:r['query_id'])
            a=photos[rows[epoch%len(rows)]['query_id']]
            aa.extend([a,a]); aa.extend(bank[mid,rng.choice(views,2,replace=False)])
            yy.extend([int(mid)]*4)
        other=rng.choice(np.setdiff1d(np.arange(len(bank)), ids), len(ids), replace=False)
        base_a=[];base_b=[];edits=[]; fields=[]
        for mid in other:
            j=int(rng.integers(2)); n=geo['normals'][mid,j]; m=geo['masks'][mid,j]
            pa=appearance(n[0],m[0],rng);pb=appearance(n[0],m[0],rng)
            pa,pb=select_lights(pa,pb,int(mid),j,epoch,start,p['seed'])
            va=rgb(n[0],m[0],pa);vb=rgb(n[0],m[0],pb);ve=rgb(n[1],m[1],pa)
            aa.extend(bank[mid,rng.choice(views,2,replace=False)])
            base_a.append(len(aa));aa.append(va)
            base_b.append(len(aa));aa.append(vb)
            edits.append(ve);fields.append(geo['fields'][mid,j]);yy.extend([int(mid)]*4)
        count=len(aa);aa.extend(edits)
        yield np.stack(aa), np.array(yy), np.array(base_a), np.array(base_b), np.arange(count,len(aa)), np.stack(fields)

def select_lights(pa,pb,mid,j,epoch,start,seed):
    key=f"{seed}:{mid}:{j}:{epoch}:{start}:light_admission".encode()
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
    choices=POLICIES[POLICY_ARM][mid,j]
    out=[]
    for params,u in zip([pa,pb],rng.random(2)):
        color,_,ambient,diffuse=params
        light=POLICIES['lights'][choices[min(int(u*len(choices)),len(choices)-1)]]
        out.append((color,light,ambient,diffuse))
    return out

def weighted_mse(error, weights):
    return (error.square()*weights[...,None]).sum()/(weights.sum().clamp_min(1e-8)*error.shape[-1])

def auxiliary(pred, fa, fb, fe, fields, geo, arm):
    g=torch.as_tensor(fields, device='cuda', dtype=torch.float32)
    a,b,e=pred[fa],pred[fb],pred[fe]
    fg=torch.maximum(g[:,0,...,4],g[:,1,...,4])
    if arm=='B':
        mean=torch.as_tensor(geo['mean'],device='cuda')
        scale=torch.as_tensor(geo['scale'],device='cuda')
        x=torch.cat([a,b,e]);target=torch.cat([g[:,0],g[:,0],g[:,1]])
        weights=torch.cat([g[:,0,...,4],g[:,0,...,4],g[:,1,...,4]])
        return weighted_mse(x-(target-mean)/scale,weights)
    if arm=='C':
        delta=g[:,1]-g[:,0]
        active=(torch.linalg.vector_norm(delta,dim=-1)>1e-5).float()
        changed=active*fg; unchanged=(1-active)*fg
        nc=changed.sum((1,2),keepdim=True);nu=unchanged.sum((1,2),keepdim=True)
        both=(nc>0)&(nu>0)
        balanced=torch.where(both,.5*changed/nc.clamp_min(1e-8)+.5*unchanged/nu.clamp_min(1e-8),fg/fg.sum((1,2),keepdim=True).clamp_min(1e-8))
        scale=torch.as_tensor(geo['delta_scale'],device='cuda')
        geometry=weighted_mse((e-a)-delta/scale,balanced)
        nuisance=weighted_mse(b-a,g[:,0,...,4])
        return .5*(geometry+nuisance)
    return pred.sum()*0

def scalar_ms(z,pairs):
    sim=F.normalize(z.detach(),dim=1)@F.normalize(z.detach(),dim=1).T
    pa,pi,na,ni=pairs;values=[]
    for i in range(len(z)):
        ps=sim[i,pi[pa==i]];ns=sim[i,ni[na==i]]
        values.append(torch.logsumexp(torch.cat([ps.new_zeros(1),-2*(ps-.5)]),0)/2+
                      torch.logsumexp(torch.cat([ns.new_zeros(1),50*(ns-.5)]),0)/50)
    return torch.stack(values).mean()

def checkpoint(model,head,arm,p,extra):
    names={n for n,v in model.named_parameters() if v.requires_grad}
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if k in names}
    torch.save(dict(trainable_state=state,geometry_head=head.state_dict(),
                    protocol_sha256=parent.sha(HERE/'protocol.json'),extra=extra),HERE/f'{arm}_final.pt')
    return parent.sha(HERE/f'{arm}_final.pt')

def train(arm,p,bank,photos,geo,clock,previous_durations):
    parent.seed(p['seed'])
    model=parent.Model().cuda()
    cp=torch.load(PARENT/'source_final.pt',map_location='cpu',weights_only=True)
    names={n for n,v in model.named_parameters() if v.requires_grad}
    assert set(cp['trainable_state'])==names
    model.load_state_dict(cp['trainable_state'],strict=False)
    head=nn.Sequential(nn.Linear(1024,128),nn.GELU(),nn.Linear(128,5)).cuda()
    parameters=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW([{'params':parameters,'lr':6e-5},{'params':head.parameters(),'lr':.001}],weight_decay=9.5e-9)
    steps=math.ceil(len({r['model_id'] for r in p['train_records']})/8)
    schedule=torch.optim.lr_scheduler.LinearLR(opt,start_factor=1,end_factor=.2,total_iters=steps*20)
    loss_fn=parent.losses.MultiSimilarityLoss(alpha=2,beta=50,base=.5)
    miner=parent.miners.MultiSimilarityMiner(epsilon=.1)
    history=[];timings=[];step=0;zeros=0;begin=time.monotonic()
    try:
        for epoch in range(20):
            model.train();head.train();losses=[]; tick_epoch=time.monotonic()
            for a,y,fa,fb,fe,fields in batches(p,bank,photos,geo,epoch):
                check(); tick=time.monotonic();opt.zero_grad(set_to_none=True)
                x=parent.input_tensor(a,True);labels=torch.as_tensor(y,device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    feats,token=model.features(x)
                    z=model.aggregation((feats[:len(y)],token[:len(y)])).float()
                    pred=head(feats.permute(0,2,3,1)).float()
                pairs=miner(z,labels); lid=loss_fn(z,labels,pairs)
                aux=auxiliary(pred,fa,fb,fe,fields,geo,arm);loss=lid+.1*aux
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                if step==0:
                    assert float((scalar_ms(z,pairs)-lid).abs())<1e-5
                loss.backward()
                grad={name:float(sum(v.grad.abs().sum().item() for v in part.parameters() if v.grad is not None))
                      for name,part in [('last_block',model.backbone.blocks[-1]),('aggregation',model.aggregation),('geometry_head',head)]}
                if not all(math.isfinite(v) for v in grad.values()):raise FloatingPointError('Nonfinite gradient')
                zeros=zeros+1 if grad['last_block']==0 else 0
                if zeros>=5:raise RuntimeError('Five consecutive zero-gradient batches')
                if step==0:
                    assert all(v.grad is None for v in model.backbone.parameters() if not v.requires_grad)
                    assert grad['last_block']>0 and grad['aggregation']>0
                    assert grad['geometry_head']==0
                    dump(f'{arm}_implementation_check.json',dict(gradients=grad,unit_norm_error=float((z.norm(dim=1)-1).abs().max()),
                       ms_reference_error=float((scalar_ms(z,pairs)-lid).abs()),frozen_gradients_absent=True,identity_images=len(y),geometry_images=len(fe),
                       virtual_geometries_excluded_from_identity_loss=True,inference_truth_input=False))
                nn.utils.clip_grad_norm_(parameters+list(head.parameters()),1.)
                opt.step();schedule.step();torch.cuda.synchronize()
                timings.append(time.monotonic()-tick);step+=1
                losses.append([float(lid.detach()),float(aux.detach()),float(loss.detach())])
                if step==8:
                    remaining_arms=4-len(previous_durations)
                    estimate=np.median(timings)*steps*20*remaining_arms*1.3+420*remaining_arms
                    dump(f'{arm}_timing_check.json',dict(first8_seconds=timings,estimated_remaining_with_margin=float(estimate),
                       remaining_budget=END-time.monotonic(),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                    if estimate>END-time.monotonic():
                        raise RuntimeError('Complete four-arm configuration cannot fit registered budget; do not shorten epochs')
            item=dict(epoch=epoch+1,identity_loss=float(np.mean(losses,0)[0]),auxiliary_loss=float(np.mean(losses,0)[1]),
                      total_loss=float(np.mean(losses,0)[2]),steps=step,epoch_seconds=time.monotonic()-tick_epoch)
            history.append(item);dump(f'{arm}_history.json',history)
            print(json.dumps(dict(arm=arm,**item,combined_seconds=time.monotonic()-clock)),flush=True)
        checksum=checkpoint(model,head,arm,p,dict(history=history,initial_checkpoint=p['initial_checkpoint_sha256']))
        return model,checksum,time.monotonic()-begin
    except Exception:
        dump(f'{arm}_incomplete.json',dict(epochs_completed=len(history),steps=step,history=history,error=traceback.format_exc()))
        raise

def evaluate(model,p,bank,photos,arm):
    check(); templates=parent.extract(model,bank.reshape(-1,224,224,3)).reshape(438,42,-1)
    rows=p['records'];q=parent.extract(model,np.stack([photos[r['query_id']] for r in rows]))
    q=unit(q);g=unit(templates.mean(1));ix={m:i for i,m in enumerate(p['gallery_model_ids'])}
    train=np.array([i for i,r in enumerate(rows) if r['outer_fold']!=3])
    target=g[[ix[rows[i]['model_id']] for i in train]]
    qt=q[train];alpha=np.linalg.solve((qt@qt.T).astype('f8')+10*np.eye(len(train)),(target-qt).astype('f8'))
    mapped=unit(q+((q@qt.T)@alpha).astype('f4'))
    scores=parent.score(mapped,templates);direct=parent.score(q,templates)
    np.savez_compressed(HERE/f'{arm}_scores.npz',scores=scores,direct=direct,query_ids=np.array([r['query_id'] for r in rows]),gallery_ids=np.array(p['gallery_model_ids']))
    np.savez_compressed(HERE/f'{arm}_alignment.npz',q=q,mean_CAD=g,train_indices=train,alpha=alpha)
    ev=np.array([i for i,r in enumerate(rows) if r['outer_fold']==3]);truth=np.array([ix[rows[i]['model_id']] for i in ev])
    result=dict(primary_ridge=parent.macro(parent.rank(scores[ev],truth),[rows[i] for i in ev]),
                secondary_direct=parent.macro(parent.rank(direct[ev],truth),[rows[i] for i in ev]),
                evaluation_queries=len(ev),training_scores_are_in_sample_not_cv_results=True)
    dump(f'{arm}_retrieval_diagnostic.json',result)
    del templates;gc.collect();return result

def main():
    global END, POLICY_ARM, POLICIES
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'));torch.set_num_threads(8)
    status=dict(status='preparing',completed_arms=[],manuscript_ready=False,original_contribution_established=False)
    try:
        assert parent.sha(PARENT/'source_final.pt')==p['initial_checkpoint_sha256']
        train_groups={r['group_id'] for r in p['train_records']};ev_groups={r['group_id'] for r in p['evaluation_records']}
        assert not train_groups&ev_groups
        assert len(p['train_records'])==391 and len(p['evaluation_records'])==97
        with np.load(ROOT/'modules/geometry_cache/geometry_bank.npz') as z:geo={k:z[k] for k in z.files}
        with np.load(HERE/'lighting_policies.npz') as z:POLICIES={k:z[k] for k in z.files}
        assert geo['cad_ids'].tolist()==p['gallery_model_ids']
        assert parent.sha(ROOT/'modules/geometry_cache/geometry_bank.npz')=='03a70891798bcffee21b84c93f32fce899d40936a0f28e803cb0930d26ed821f'
        assert parent.sha(HERE/'lighting_policies.npz')==json.loads((HERE/'policy_checks.json').read_text())['sha256']
        photos={r['query_id']:parent.image(ROOT/r['repaired_path']) for r in p['records']}
        bank=np.empty((438,42,224,224,3),dtype=np.uint8)
        for i,mid in enumerate(p['gallery_model_ids']):
            for v in range(42):bank[i,v]=parent.image(ROOT/'modules/cache/renders_sphere42'/mid/f'view_{v:02d}.png')
        dump('isolation_check.json',dict(training_groups=len(train_groups),evaluation_groups=len(ev_groups),overlap=[],
                  train_photo_count=391,evaluation_photo_count=97,all_gallery_CADs_synthetic_access=True,
                  protocol_sha256=parent.sha(HERE/'protocol.json'),script_sha256=parent.sha(__file__)))
        clock=time.monotonic();END=clock+3600;parent.DEADLINE=END;durations=[]
        for arm in ['R','Q','O','G']:
            POLICY_ARM=arm
            status['status']='running_'+arm;dump('run_status.json',status)
            model,checksum,duration=train(arm,p,bank,photos,geo,clock,durations)
            metrics=evaluate(model,p,bank,photos,arm)
            durations.append(duration);status['completed_arms'].append(dict(arm=arm,checkpoint_sha256=checksum,metrics=metrics))
            status['combined_wall_seconds']=time.monotonic()-clock;dump('run_status.json',status)
            del model;gc.collect();torch.cuda.empty_cache()
        status['status']='complete_four_arm_scores_pending_independent_evaluation';dump('run_status.json',status)
        print(json.dumps(status),flush=True)
    except Exception:
        status.update(status='incomplete_execution',error=traceback.format_exc());dump('run_status.json',status)
        print(json.dumps(status),flush=True);raise

if __name__=='__main__':main()
