"""Complete fixed-score evaluation. Truth is used after assignment only."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import csv,gzip,hashlib,itertools,json,time,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent;PUB=HERE.parent
sys.path.insert(0,str(PUB/'assignment'))
from assignment import assign
PRIMARY=20261005
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def write(name,rows):
    if not rows:return
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def z(a):return (a-a.mean(1,keepdims=True))/np.maximum(a.std(1,keepdims=True),1e-8)
def rank_matrix(a):
    return np.argsort(np.argsort(-a,axis=1,kind='stable'),axis=1,kind='stable')+1
def id_values(rr):
    byq=defaultdict(list);m={}
    for r in rr:byq[r['query_id']].append(float(r['correct']));m[r['query_id']]=r['true_cad']
    bym=defaultdict(list)
    for q,v in byq.items():bym[m[q]].append(float(np.mean(v)))
    return {i:float(np.mean(v)) for i,v in bym.items()}
def stats(rr):
    a=id_values(rr);cases=defaultdict(list);batches=defaultdict(list);dup=defaultdict(list)
    for r in rr:cases[r['case_id']].append(r);batches[r['batch_id']].append(r['case_id'])
    exact={c:all(x['correct'] for x in rs) for c,rs in cases.items()}
    duplicates=[len(rs)-len({r['prediction'] for r in rs}) for rs in cases.values()]
    return dict(object_macro_accuracy=float(np.mean(list(a.values()))),identities=len(a),queries=len({r['query_id'] for r in rr}),
                observations=len(rr),case_count=len(cases),unique_batches=len(batches),
                whole_batch_exact=float(np.mean(list(exact.values()))),
                unique_batch_exact_macro=float(np.mean([np.mean([exact[c] for c in set(cs)]) for cs in batches.values()])),
                mean_duplicate_assignments=float(np.mean(duplicates)))
def paired(rr,ss,qmap):
    a=id_values(rr);b=id_values(ss);assert set(a)==set(b)
    parents={r['batch_id']:r['batch_id'] for r in rr}
    def find(v):
        while parents[v]!=v:parents[v]=parents[parents[v]];v=parents[v]
        return v
    gs=defaultdict(set);mb={}
    for r in rr:
        gs[qmap[r['query_id']]['group_id']].add(r['batch_id'])
        assert mb.setdefault(r['true_cad'],r['batch_id'])==r['batch_id']
    for bs in gs.values():
        bs=sorted(bs)
        for x in bs[1:]:parents[find(x)]=find(bs[0])
    units=defaultdict(list)
    for m in sorted(a):units[find(mb[m])].append(m)
    vals=list(units.values());dd=np.array([sum(a[m]-b[m] for m in ids) for ids in vals]);nn=np.array([len(ids) for ids in vals])
    rng=np.random.default_rng(20261006);idx=rng.integers(len(vals),size=(10000,len(vals)))
    boot=dd[idx].sum(1)/nn[idx].sum(1)
    return dict(difference=float(dd.sum()/nn.sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                bootstrap_units=len(vals),objects=len(a),replicates=10000,conditional='fixed out-of-fold models and main batch partition')
def true_ranks(score,truth):
    gt=score[np.arange(len(truth)),truth];ix=np.arange(score.shape[1])
    return 1+(score>gt[:,None]).sum(1)+((score==gt[:,None])&(ix[None,:]<truth[:,None])).sum(1)
def macro_rank(ranks,records):
    bym=defaultdict(list)
    for rr,r in zip(ranks,records):bym[r['model_id']].append(int(rr))
    return dict(identities=len(bym),queries=len(records),**{('top1' if k==1 else f'recall_at_{k}'):float(np.mean([np.mean(np.array(v)<=k) for v in bym.values()])) for k in [1,5,10,20,100]},
                mrr=float(np.mean([np.mean(1/np.array(v)) for v in bym.values()])))
def combine(p,kind,seed=None):
    records=p['records'];cids=p['gallery_model_ids'];out=np.zeros((488,438));direct=out.copy();covered=np.zeros(488,int)
    for f in range(5):
        folder=HERE/(f'source_f{f}' if kind=='source' else f'R_s{seed}_f{f}')
        with np.load(folder/'scores.npz') as s:
            assert s['query_ids'].tolist()==[r['query_id'] for r in records] and s['gallery_ids'].tolist()==cids
            ii=np.array([i for i,r in enumerate(records) if r['outer_fold']==f])
            out[ii]=s['scores'][ii];direct[ii]=s['direct'][ii];covered[ii]+=1
    assert np.all(covered==1) and np.isfinite(out).all() and np.isfinite(direct).all()
    np.savez_compressed(HERE/(f'oof_source.npz' if kind=='source' else f'oof_R_s{seed}.npz'),scores=out,direct=direct,query_ids=np.array([r['query_id'] for r in records]),gallery_ids=np.array(cids))
    return out,direct
def matrices(h,src,srcdirect,learned,direct,ii,jj):
    subset=lambda a:a[ii][:,jj]
    s=subset(h['sdf']);r=subset(h['ridge']);a=subset(learned);d=subset(direct)
    ar=rank_matrix(a);sr=rank_matrix(s)
    mats={f'history_{k}':subset(v) for k,v in h.items()}
    mats.update(history_matched_fusion=.5*(z(r)+z(s)),source_ridge=subset(src),
                source_fusion=.5*(z(subset(src))+z(s)),source_direct_fusion=.5*(z(subset(srcdirect))+z(s)),
                R_direct=d,R_ridge=a,R_direct_fusion=.5*(z(d)+z(s)),R_fusion=.5*(z(a)+z(s)),
                R_rrf60=.5*(1/(60+ar)+1/(60+sr)),R_borda=-.5*(ar+sr))
    return mats
def methods(seed,primary):
    if seed!=primary:return [(m,s) for m in ['R_direct','R_ridge','R_direct_fusion','R_fusion'] for s in ['independent','global']]+[('R_fusion','greedy'),('R_rrf60','global'),('R_borda','global')]
    return [(f'history_{m}',s) for m in ['cnos_rgb','sdf','ridge','z_fusion','rrf60','borda','supervised'] for s in ['independent','global']]+[('history_z_fusion','greedy')]+[(m,s) for m in ['history_matched_fusion','source_ridge','source_fusion','source_direct_fusion','R_direct','R_ridge','R_direct_fusion','R_fusion'] for s in ['independent','global']]+[('history_matched_fusion','greedy'),('R_fusion','greedy'),('R_rrf60','global'),('R_borda','global')]

def main():
    if (HERE/'ANALYSIS_COMPLETE.json').exists():raise FileExistsError('Do not overwrite completed results')
    start=time.monotonic();p=read(HERE/'protocol.json');records=p['records'];qm={r['query_id']:r for r in records}
    qi={q:i for i,q in enumerate(qm)};ci={m:i for i,m in enumerate(p['gallery_model_ids'])};sourcekind=read(PUB/'data_groups/protocol.json')['source_kind_by_query']
    assert set(sourcekind)==set(qm)
    with np.load(PUB/'score_cache/scores_438.npz') as f:h={k:f[k].astype('f8') for k in ['cnos_rgb','sdf','ridge','z_fusion','rrf60','borda']}
    with np.load(PUB/'inventories/fusion_scores.npz') as f:h['supervised']=f['scores'].astype('f8')
    src,srcdirect=combine(p,'source')
    defs={x['batch_id']:x for x in gz(PUB/'inventories/batch_definitions.json.gz')}
    cases=[c for c in gz(PUB/'inventories/batch_cases.json.gz') if c['scenario']=='closed']
    shared=[]
    for cap in [10,20,40]:shared.append({m for d in defs.values() if d['layout']=='random' and d['cohort']=='waxup' and d['seed']==PRIMARY and d['capacity']==cap and d['full_capacity_batch'] for m in d['candidate_cad_ids']})
    shared=set.intersection(*shared);(HERE/'shared_capacity_ids.json').write_text(json.dumps(sorted(shared),indent=2))
    summaries=[];folds=[];cis=[];absolute=[];changes=[];capacity_same=[];sources=[];retrieval=[];queryranks=[];coverage=[];maincache={}
    fields=['training_seed','method','strategy','layout','cohort','capacity','layout_seed','fold','batch_id','case_id','full','query_id','true_cad','prediction','correct','group_id','assigned_score']
    with gzip.open(HERE/'predictions.csv.gz','wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for seed in p['training_seeds']:
            learned,direct=combine(p,'R',seed);scopes=defaultdict(list);met=methods(seed,p['primary_seed'])
            for c in cases:
                de=defs[c['batch_id']];qs=c['query_ids'];cads=de['candidate_cad_ids'];ii=[qi[q] for q in qs];jj=[ci[m] for m in cads]
                assert len(qs)==len(cads) and all(qm[q]['outer_fold']==de['fold'] for q in qs)
                mats=matrices(h,src,srcdirect,learned,direct,ii,jj)
                for m,st in met:
                    output=assign(mats[m],cads,de['counts'],st)
                    for pos,(q,y) in enumerate(zip(qs,output['predictions'])):
                        rr=dict(training_seed=seed,method=m,strategy=st,layout=de['layout'],cohort=de['cohort'],capacity=de['capacity'],layout_seed=de['seed'],
                                fold=de['fold'],batch_id=de['batch_id'],case_id=c['case_id'],full=de['full_capacity_batch'],query_id=q,true_cad=qm[q]['model_id'],
                                prediction=y,correct=y==qm[q]['model_id'],group_id=qm[q]['group_id'],assigned_score=float(mats[m][pos,cads.index(y)]))
                        writer.writerow(rr);scopes[(de['layout'],de['cohort'],de['capacity'],de['seed'],m,st)].append(rr)
            for key,rr in scopes.items():
                meta=dict(training_seed=seed,**dict(zip(['layout','cohort','capacity','layout_seed','method','strategy'],key)))
                for pop in ['all','full','tail']:
                    rows=[r for r in rr if pop=='all' or bool(r['full'])==(pop=='full')]
                    if not rows:continue
                    summaries.append(dict(**meta,population=pop,**stats(rows)))
                    if meta['layout_seed']==PRIMARY:
                        for fold in sorted({r['fold'] for r in rows}):
                            folds.append(dict(**meta,population=pop,fold=fold,**stats([r for r in rows if r['fold']==fold])))
                        if meta['layout']=='random' and meta['cohort']=='waxup' and meta['capacity'] in [10,20,40] and pop=='full':
                            same=[r for r in rows if r['true_cad'] in shared]
                            stat=stats(same)
                            capacity_same.append(dict(**meta,population='same_ids_within_original_full_batches',**{k:stat[k] for k in ['object_macro_accuracy','identities','queries','observations']}))
                        if meta['cohort'] in ['all438_mixed','waxup'] and meta['capacity']==20:
                            for kind in sorted(set(sourcekind.values())):
                                subset=[r for r in rows if sourcekind[r['query_id']]==kind]
                                if subset:
                                    stat=stats(subset)
                                    sources.append(dict(**meta,population=pop,source_kind=kind,groups=len({r['group_id'] for r in subset}),**{k:stat[k] for k in ['object_macro_accuracy','identities','queries','observations']}))
                    if meta['cohort']=='waxup' and meta['capacity']==20 and meta['layout_seed']==PRIMARY and pop=='full':
                        maincache[(seed,meta['layout'],meta['method'],meta['strategy'])]=rows
            covered={r['query_id'] for rr in scopes.values() for r in rr if r['method']=='R_fusion' and r['strategy']=='global'}
            assert covered==set(qm)
            coverage.append(dict(training_seed=seed,photos=len(covered),identities=len({qm[q]['model_id'] for q in covered}),all_488_retained=True))
            truth=np.array([ci[r['model_id']] for r in records])
            fullscores={'R_ridge':learned,'R_direct':direct,'R_fusion':.5*(z(learned)+z(h['sdf']))}
            if seed==p['primary_seed']:fullscores.update(source_ridge=src,source_fusion=.5*(z(src)+z(h['sdf'])),**{'history_'+k:v for k,v in h.items()})
            for method,scores in fullscores.items():
                ranks=true_ranks(scores,truth)
                for fold in [-1,0,1,2,3,4]:
                    ii=[i for i,r in enumerate(records) if fold==-1 or r['outer_fold']==fold]
                    retrieval.append(dict(training_seed=seed,method=method,fold=fold,**macro_rank(ranks[ii],[records[i] for i in ii])))
                for i,r in enumerate(records):queryranks.append(dict(training_seed=seed,method=method,query_id=r['query_id'],model_id=r['model_id'],group_id=r['group_id'],fold=r['outer_fold'],source_kind=sourcekind[r['query_id']],rank=int(ranks[i])))
            del scopes
            print(json.dumps(dict(seed_complete=seed,wall_seconds=time.monotonic()-start)),flush=True)
    for (seed,layout,method,strategy),rows in maincache.items():
        zero=[dict(r,correct=False) for r in rows]
        result=paired(rows,zero,qm)
        absolute.append(dict(training_seed=seed,layout=layout,method=method,strategy=strategy,accuracy=result['difference'],
                             **{k:v for k,v in result.items() if k!='difference'}))
    development_excluded=[]
    for (seed,layout,method,strategy),rows in maincache.items():
        subset=[r for r in rows if r['fold']!=3]
        development_excluded.append(dict(training_seed=seed,layout=layout,method=method,strategy=strategy,
                                         excluded_fold=3,interpretation='retrospective sensitivity, not a new independent test',**stats(subset)))
    comparators=[('history_matched_fusion','global'),('history_z_fusion','global'),('history_supervised','global'),('history_rrf60','global'),
                 ('source_fusion','global'),('R_direct_fusion','global'),('R_ridge','global'),('history_sdf','global'),('R_fusion','independent'),('R_fusion','greedy'),('R_rrf60','global'),('R_borda','global')]
    for seed in p['training_seeds']:
        for layout in ['random','hard']:
            a=maincache[(seed,layout,'R_fusion','global')]
            for method,strategy in comparators:
                b=maincache.get((seed,layout,method,strategy),maincache.get((p['primary_seed'],layout,method,strategy)))
                if b is None:continue
                result=paired(a,b,qm);cis.append(dict(training_seed=seed,layout=layout,method='R_fusion',strategy='global',comparator=method,comparator_strategy=strategy,**result))
                # Counting corrections/new errors is descriptive; rotations are averaged by identity.
                aa=id_values(a);bb=id_values(b)
                changes.append(dict(training_seed=seed,layout=layout,comparator=method,comparator_strategy=strategy,
                     improved_ids=sum(aa[m]>bb[m]+1e-12 for m in aa),worsened_ids=sum(aa[m]<bb[m]-1e-12 for m in aa),unchanged_ids=sum(abs(aa[m]-bb[m])<=1e-12 for m in aa),
                     total_gain=sum(max(0,aa[m]-bb[m]) for m in aa),total_loss=sum(max(0,bb[m]-aa[m]) for m in aa)))
    sensitivity=[]
    for seed,layout,method,strategy in sorted({(r['training_seed'],r['layout'],r['method'],r['strategy']) for r in summaries if r['cohort']=='waxup' and r['capacity']==20 and r['population']=='full'}):
        rr=[r for r in summaries if r['training_seed']==seed and r['layout']==layout and r['method']==method and r['strategy']==strategy and r['cohort']=='waxup' and r['capacity']==20 and r['population']=='full']
        assert len(rr)==25
        v=[r['object_macro_accuracy'] for r in rr]
        sensitivity.append(dict(training_seed=seed,layout=layout,method=method,strategy=strategy,mean=float(np.mean(v)),minimum=min(v),maximum=max(v),layouts=25,independent_samples=False))
    for name,rr in [('all_results.csv',summaries),('fold_results.csv',folds),('paired_intervals.csv',cis),('absolute_intervals.csv',absolute),('error_changes.csv',changes),('capacity_same_ids.csv',capacity_same),
                    ('source_analysis.csv',sources),('retrieval_results.csv',retrieval),('retrieval_per_query.csv',queryranks),('layout_sensitivity.csv',sensitivity),('coverage.csv',coverage),
                    ('development_fold_excluded_results.csv',development_excluded)]:write(name,rr)
    primary=[r for r in summaries if r['training_seed']==p['primary_seed'] and r['layout_seed']==PRIMARY and r['cohort']=='waxup' and r['capacity']==20 and r['population']=='full']
    write('primary_results.csv',primary)
    result=dict(status='all_registered_scores_analyzed_pending_independent_verification',protocol_sha256=sha(HERE/'protocol.json'),training_seeds=p['training_seeds'],
        summary_rows=len(summaries),paired_intervals=len(cis),shared_capacity_identities=len(shared),coverage=coverage,primary=primary,
        manuscript=False,original_algorithm_claim=False,simulated_batches=True,retrospective=True,wall_seconds=time.monotonic()-start)
    (HERE/'ANALYSIS_COMPLETE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='primary'},ensure_ascii=False))

if __name__=='__main__':main()
