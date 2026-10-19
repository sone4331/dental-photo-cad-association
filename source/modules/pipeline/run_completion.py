"""Exact fixed R across folds/seeds; existing source checkpoint is a control."""
import os
os.environ['XFORMERS_DISABLED']='1'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
os.environ['OMP_NUM_THREADS']='8'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc,importlib.util,json,math,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'modules/adaptation'
spec=importlib.util.spec_from_file_location('adaptation_training',OLD/'controlled_train.py')
ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
parent=ctl.parent
END=float('inf')
def dump(name,x):
    path=HERE/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def check():
    if time.monotonic()>END:raise RuntimeError('Fixed completion budget exhausted; incomplete state retained')
def load_model(path):
    parent.seed(20261006);model=parent.Model().cuda()
    cp=torch.load(path,map_location='cpu',weights_only=True)
    names={n for n,v in model.named_parameters() if v.requires_grad}
    assert set(cp['trainable_state'])==names
    model.load_state_dict(cp['trainable_state'],strict=False)
    return model

def extract_all(model,bank,photos,records):
    t=time.monotonic()
    templates=parent.extract(model,bank.reshape(-1,224,224,3)).reshape(len(bank),42,-1)
    queries=parent.extract(model,np.stack([photos[r['query_id']] for r in records]))
    assert np.max(np.abs(np.linalg.norm(templates,axis=-1)-1))<1e-4
    assert np.max(np.abs(np.linalg.norm(queries,axis=-1)-1))<1e-4
    return templates,queries,time.monotonic()-t

def alignment_and_scores(templates,q,records,cids,fold,folder):
    check();begin=time.monotonic();folder.mkdir(exist_ok=True,parents=True)
    ix={m:i for i,m in enumerate(cids)}
    train=np.array([i for i,r in enumerate(records) if r['outer_fold']!=fold])
    ev=np.array([i for i,r in enumerate(records) if r['outer_fold']==fold])
    q=ctl.unit(q);g=ctl.unit(templates.mean(1));qt=q[train]
    target=g[[ix[records[i]['model_id']] for i in train]]
    alpha=np.linalg.solve((qt@qt.T).astype('f8')+10*np.eye(len(train)),(target-qt).astype('f8'))
    mapped=ctl.unit(q+((q@qt.T)@alpha).astype('f4'))
    scores=parent.score(mapped,templates);direct=parent.score(q,templates)
    assert np.isfinite(scores).all() and np.isfinite(direct).all()
    np.savez_compressed(folder/'scores.npz',scores=scores,direct=direct,query_ids=np.array([r['query_id'] for r in records]),gallery_ids=np.array(cids),evaluation_indices=ev)
    np.savez_compressed(folder/'alignment.npz',q=q,mean_CAD=g,train_indices=train,alpha=alpha)
    sample_q=ev[:6];sample_c=np.linspace(0,len(cids)-1,8,dtype=int)
    np.savez_compressed(folder/'score_rebuild_sample.npz',q=q[sample_q],mapped=mapped[sample_q],templates=templates[sample_c],
                        scores=scores[sample_q][:,sample_c],direct=direct[sample_q][:,sample_c],query_indices=sample_q,cad_indices=sample_c)
    truth=np.array([ix[records[i]['model_id']] for i in ev])
    metrics={name:parent.macro(parent.rank(s[ev],truth),[records[i] for i in ev]) for name,s in [('ridge',scores),('direct',direct)]}
    return dict(metrics=metrics,score_seconds=time.monotonic()-begin,score_sha256=parent.sha(folder/'scores.npz'),evaluation_queries=len(ev))

def main():
    global END
    if (HERE/'GPU_COMPLETE.json').exists():raise FileExistsError('Completed runs are immutable')
    p=json.loads((HERE/'protocol.json').read_text());torch.set_num_threads(8)
    start=time.monotonic();status=dict(status='preflight',completed=[],new_training_runs=0)
    try:
        assert parent.sha(ctl.PARENT/'source_final.pt')==p['initial_checkpoint_sha256']
        for rel,h in p['source_hashes'].items():assert parent.sha(ROOT/'modules'/rel)==h,rel
        pm=json.loads((HERE/'photo_input_manifest.json').read_text())
        for r in pm:assert parent.sha(ROOT/r['path'])==r['sha256'],r['query_id']
        photos={r['query_id']:parent.image(ROOT/r['repaired_path']) for r in p['records']}
        bank=np.empty((438,42,224,224,3),np.uint8)
        for i,mid in enumerate(p['gallery_model_ids']):
            for v in range(42):bank[i,v]=parent.image(ROOT/'modules/cache/renders_sphere42'/mid/f'view_{v:02d}.png')
        with np.load(ROOT/'modules/geometry_cache/geometry_bank.npz') as z:geo={k:z[k] for k in z.files}
        with np.load(OLD/'lighting_policies.npz') as z:ctl.POLICIES={k:z[k] for k in z.files}
        assert geo['cad_ids'].tolist()==p['gallery_model_ids']
        assert parent.sha(ROOT/'modules/geometry_cache/geometry_bank.npz')=='03a70891798bcffee21b84c93f32fce899d40936a0f28e803cb0930d26ed821f'
        assert parent.sha(OLD/'lighting_policies.npz')==json.loads((OLD/'policy_checks.json').read_text())['sha256']
        preparation=time.monotonic()-start;END=time.monotonic()+p['budget']['gpu_stage_wall_seconds'];parent.DEADLINE=END;ctl.END=END;ctl.POLICY_ARM='R'
        status.update(preparation_seconds=preparation,status='source_control_feature_extraction');dump('run_status.json',status)
        model=load_model(ctl.PARENT/'source_final.pt')
        templates,q,extract_seconds=extract_all(model,bank,photos,p['records'])
        np.savez_compressed(HERE/'source_descriptors.npz',templates=templates,q=q)
        for fold in range(5):
            result=alignment_and_scores(templates,q,p['records'],p['gallery_model_ids'],fold,HERE/f'source_f{fold}')
            dump(f'source_f{fold}/metrics.json',result)
        del model,templates,q;gc.collect();torch.cuda.empty_cache()
        dump('source_control.json',dict(feature_seconds=extract_seconds,source_checkpoint_sha256=p['initial_checkpoint_sha256'],
             real_backbone_adaptation=False,real_labels_used_for_train_side_ridge=True,source_descriptors_sha256=parent.sha(HERE/'source_descriptors.npz')))
        durations=[]
        for job in p['execution_order']:
            check();seed=job['seed'];fold=job['fold'];name=f'R_s{seed}_f{fold}';folder=HERE/name
            if folder.exists():raise FileExistsError('No silent resume or outcome overwrite: '+name)
            folder.mkdir();pp=dict(p,seed=seed,outer_fold=fold,train_records=[r for r in p['records'] if r['outer_fold']!=fold],evaluation_records=[r for r in p['records'] if r['outer_fold']==fold])
            (folder/'protocol.json').write_text(json.dumps(pp,ensure_ascii=False,indent=2))
            status.update(status='running_'+name,current_job=job);dump('run_status.json',status)
            ctl.HERE=folder
            if job['reuse']:
                # Reuse scores/weights verbatim, rather than rerunning the already explored fold.
                shutil.copy2(OLD/'R_scores.npz',folder/'scores.npz')
                shutil.copy2(OLD/'R_alignment.npz',folder/'alignment.npz')
                shutil.copy2(OLD/'R_final.pt',folder/'R_final.pt')
                shutil.copy2(OLD/'R_history.json',folder/'R_history.json')
                result=dict(reused=True,score_sha256=parent.sha(folder/'scores.npz'),checkpoint_sha256=parent.sha(folder/'R_final.pt'),training_seconds=0.)
            else:
                begin=time.monotonic()
                model,cp,duration=ctl.train('R',pp,bank,photos,geo,begin,[])
                durations.append(duration)
                templates,q,feature_seconds=extract_all(model,bank,photos,p['records'])
                result=alignment_and_scores(templates,q,p['records'],p['gallery_model_ids'],fold,folder)
                result.update(reused=False,training_seconds=duration,feature_seconds=feature_seconds,checkpoint_sha256=cp,
                              peak_gpu_bytes=torch.cuda.max_memory_allocated())
                del model,templates,q;gc.collect();torch.cuda.empty_cache();status['new_training_runs']+=1
            dump(name+'/metrics.json',result)
            status['completed'].append(dict(name=name,seed=seed,fold=fold,**result))
            status['gpu_stage_seconds']=time.monotonic()-start-preparation;dump('run_status.json',status)
            print(json.dumps(dict(completed=name,reused=result['reused'],new_training_runs=status['new_training_runs'],elapsed=status['gpu_stage_seconds'])),flush=True)
            if durations and len(status['completed'])==1:
                remaining=sum(not j['reuse'] for j in p['execution_order'][1:])
                estimate=remaining*(durations[-1]+result.get('feature_seconds',0)+result.get('score_seconds',0))*1.3
                dump('full_schedule_cost_check.json',dict(estimated_remaining_seconds=estimate,remaining_budget=END-time.monotonic(),planned_new_runs=14))
                if estimate>END-time.monotonic():raise RuntimeError('Complete registered seeds/folds do not fit budget; save state, do not shorten training')
        status.update(status='complete_fixed_seed_fold_scores',gpu_stage_seconds=time.monotonic()-start-preparation,original_algorithm_claim=False,manuscript=False)
        dump('GPU_COMPLETE.json',status);dump('run_status.json',status)
        print(json.dumps(dict(status=status['status'],completed_runs=len(status['completed']),wall_seconds=status['gpu_stage_seconds'])),flush=True)
    except Exception:
        status.update(status='incomplete',error=traceback.format_exc(),wall_seconds=time.monotonic()-start)
        dump('run_status.json',status);print(json.dumps(status),flush=True);raise

if __name__=='__main__':main()
