"""End-to-end warmed batch inference on fixed inventories; no accuracy fitting."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='8'
os.environ['XFORMERS_DISABLED']='1'
import csv,gzip,hashlib,importlib.util,json,platform,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PUB=HERE.parent
spec=importlib.util.spec_from_file_location('fixed_train',HERE/'run_completion.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
sys.path.insert(0,str(PUB/'silhouette'));from geometry_baseline import descriptor
sys.path.insert(0,str(PUB/'assignment'));from assignment import assign
def sync():torch.cuda.synchronize();return time.perf_counter()
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9)
def zz(a):return (a-a.mean(1,keepdims=True))/np.maximum(a.std(1,keepdims=True),1e-8)
def loadgz(path):
    with gzip.open(path,'rt') as f:return json.load(f)
def main():
    if (HERE/'TIMING_COMPLETE.json').exists():raise FileExistsError('Do not overwrite timing measurements')
    p=json.loads((HERE/'protocol.json').read_text());torch.set_num_threads(8)
    precision=json.loads((HERE/'BATCH_PRECISION_DIAGNOSIS.json').read_text())
    assert precision['reference_batch32_max_descriptor_error']<1e-6 and precision['no_score_or_model_changes']
    definitions=loadgz(PUB/'inventories/batch_definitions.json.gz');cases=loadgz(PUB/'inventories/batch_cases.json.gz')
    chosen=[]
    for cap in [10,20,40]:
        d=min((d for d in definitions if d['fold']==3 and d['seed']==20261005 and d['capacity']==cap and d['cohort']=='waxup' and d['layout']=='random' and d['full_capacity_batch']),key=lambda d:d['batch_id'])
        c=min((c for c in cases if c['batch_id']==d['batch_id'] and c['scenario']=='closed'),key=lambda c:c['case_id']);chosen.append((d,c))
    ids=sorted({m for d,c in chosen for m in d['candidate_cad_ids']});ix={m:i for i,m in enumerate(ids)}
    begin=time.monotonic();model=run.load_model(HERE/'R_s20261006_f3/R_final.pt');torch.cuda.synchronize();model_load_seconds=time.monotonic()-begin;torch.cuda.reset_peak_memory_stats()
    gallery_begin=time.monotonic()
    bank=np.stack([run.parent.image(ROOT/'modules/cache/renders_sphere42'/m/f'view_{v:02d}.png') for m in ids for v in range(42)])
    gallery_read_seconds=time.monotonic()-gallery_begin;feature_begin=time.monotonic()
    templates=run.parent.extract(model,bank).reshape(len(ids),42,-1);torch.cuda.synchronize();gallery_feature_seconds=time.monotonic()-feature_begin;offline_seconds=time.monotonic()-begin
    with np.load(HERE/'R_s20261006_f3/alignment.npz') as a:qt=a['q'][a['train_indices']];alpha=a['alpha']
    with np.load(PUB/'cache/features/sphere42_shape.npz') as a:
        allids=[str(x) for x in a['ids']];lookup={m:i for i,m in enumerate(allids)}
        allshape=a['features'].reshape(len(allids),42,3,1024)[:,:,1].astype('f4');shape=allshape[[lookup[m] for m in ids]]
    with np.load(PUB/'score_cache/scores_438.npz') as a:old_sdf=a['sdf']
    with np.load(HERE/'R_s20261006_f3/scores.npz') as a:old_visual=a['scores']
    rows=p['records'];qm={r['query_id']:r for r in rows};qi={r['query_id']:i for i,r in enumerate(rows)};ci={m:i for i,m in enumerate(p['gallery_model_ids'])}
    output=[];checks=[]
    def one(d,c):
        qs=c['query_ids'];cids=d['candidate_cad_ids'];jj=[ix[m] for m in cids]
        t0=sync();arrays=np.stack([run.parent.image(ROOT/qm[q]['repaired_path']) for q in qs]);t1=sync()
        features=run.parent.extract(model,arrays);t2=sync()
        mapped=unit(features+((features@qt.T)@alpha).astype('f4'));visual=run.parent.score(mapped,templates[jj]);t3=sync()
        sdfq=np.stack([descriptor(ROOT/qm[q]['repaired_path'],angle) for q in qs for angle in range(0,360,15)])
        target=shape[jj].reshape(-1,1024)
        dist=(sdfq*sdfq).mean(1)[:,None]+(target*target).mean(1)[None]-2*(sdfq@target.T)/1024
        sdf=(-np.sqrt(np.maximum(dist,0))).reshape(len(qs),24,len(jj),42).max((1,3));t4=sync()
        fused=.5*(zz(visual)+zz(sdf));pred=assign(fused,cids,[1]*len(cids),'global')['predictions'];t5=sync()
        timing=dict(preprocess_ms=(t1-t0)*1000,features_ms=(t2-t1)*1000,alignment_visual_scoring_ms=(t3-t2)*1000,
                    SDF_descriptor_scoring_ms=(t4-t3)*1000,fusion_assignment_ms=(t5-t4)*1000,total_ms=(t5-t0)*1000)
        return timing,visual,sdf,pred
    for d,c in chosen:
        for _ in range(2):one(d,c)
        for rep in range(10):
            timing,v,s,pred=one(d,c);output.append(dict(capacity=d['capacity'],repeat=rep,batch_id=d['batch_id'],**timing))
        ii=[qi[q] for q in c['query_ids']];jj=[ci[m] for m in d['candidate_cad_ids']]
        expected=assign(.5*(zz(old_visual[ii][:,jj])+zz(old_sdf[ii][:,jj])),d['candidate_cad_ids'],[1]*len(jj),'global')['predictions']
        checks.append(dict(capacity=d['capacity'],visual_cached_max_error=float(np.max(abs(v-old_visual[ii][:,jj]))),
                           sdf_cached_max_error=float(np.max(abs(s-old_sdf[ii][:,jj]))),prediction_disagreements=sum(x!=y for x,y in zip(pred,expected))))
        print(json.dumps(dict(live_score_check=checks[-1])),flush=True)
        if checks[-1]['visual_cached_max_error']>=.003 or checks[-1]['sdf_cached_max_error']>=2e-4 or checks[-1]['prediction_disagreements']!=0:
            np.savez_compressed(HERE/f'timing_check_failure_c{d["capacity"]}.npz',live_visual=v,cached_visual=old_visual[ii][:,jj],live_sdf=s,cached_sdf=old_sdf[ii][:,jj],query_ids=np.array(c['query_ids']),cad_ids=np.array(d['candidate_cad_ids']))
            (HERE/'TIMING_CHECK_FAILURE.json').write_text(json.dumps(checks,indent=2))
        # Identical reference batching has independently reproduced ALL cached
        # queries exactly. Inventory-sized BF16 kernels are not numerically
        # identical. Record their drift and label changes; never replace the
        # main cached evaluation, or require favorable prediction agreement.
        assert np.isfinite(v).all() and checks[-1]['sdf_cached_max_error']<2e-4
    with (HERE/'timing_observations.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    summaries=[]
    for cap in [10,20,40]:
        for field in ['preprocess_ms','features_ms','alignment_visual_scoring_ms','SDF_descriptor_scoring_ms','fusion_assignment_ms','total_ms']:
            v=[r[field] for r in output if r['capacity']==cap]
            summaries.append(dict(capacity=cap,stage=field,measurements=len(v),mean_ms=float(np.mean(v)),P50_ms=float(np.percentile(v,50)),P95_ms=float(np.percentile(v,95)),P99_ms=float(np.percentile(v,99)),maximum_ms=max(v)))
    with (HERE/'timing_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    result=dict(status='complete_warmed_pre_cropped_batch_pipeline_timing',checks=checks,hardware=torch.cuda.get_device_name(0),GPU_memory_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        CPU_threads=8,offline_gallery_model_load_extraction_seconds=offline_seconds,gallery_CADs_prepared=len(ids),templates_prepared=len(ids)*42,
        model_load_seconds=model_load_seconds,gallery_read_resize_seconds=gallery_read_seconds,gallery_feature_seconds=gallery_feature_seconds,
        descriptor_dimension=int(templates.shape[-1]),total_model_parameters=sum(v.numel() for v in model.parameters()),trainable_model_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        stored_template_bytes=int(templates.nbytes),template_bytes_per_CAD=int(templates.nbytes//len(ids)),
        full438_template_bytes_estimate=int(templates.nbytes//len(ids)*438),full438_feature_seconds_estimate=float(gallery_feature_seconds/len(ids)*438),
        offline_subset_scaling_is_estimate_not_full_gallery_measurement=True,
        excludes=['manual photo crop','model loading','CAD rendering','template feature extraction','training','ridge fitting'],
        model_reused='primary seed fold3',warmup_repeats=2,measured_repeats=10,timed_cases=[d['batch_id'] for d,c in chosen],
        limits='one hash-fixed batch per size; warmed files/kernels; empirical small-sample quantiles, no realtime guarantee',summaries=summaries)
    result.update(BF16_batch_shape_precision_diagnostic=precision,
                  live_predictions_not_assumed_identical_to_cached=True,
                  main_evaluation_scores_unchanged=True,
                  score_drift_and_label_changes_reported=checks)
    (HERE/'TIMING_COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='summaries'}))
    # Separate offline cost sample; it cannot change any recognition score or gallery.
    import subprocess
    subprocess.run([sys.executable,str(HERE/'offline_render_cost.py')],check=True)
if __name__=='__main__':main()
