"""Rebuild assignments and aggregates from raw scores without analyze.py."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import csv,gzip,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE=Path(__file__).resolve().parent;PUB=HERE.parent
def read(path):return json.loads(Path(path).read_text())
def table(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def gz(path):
    with gzip.open(path,'rt') as f:return json.load(f)
def centered(a):
    mu=np.sum(a,axis=1,keepdims=True)/a.shape[1];dd=a-mu
    return dd/np.maximum(np.sqrt(np.sum(dd*dd,axis=1,keepdims=True)/a.shape[1]),1e-8)
def ordinal(a):
    sorted_columns=np.argsort(-a,axis=1,kind='stable');out=np.empty_like(sorted_columns)
    np.put_along_axis(out,sorted_columns,np.arange(1,a.shape[1]+1)[None],axis=1);return out
def build(method,h,source,learned,ii,jj):
    cut=lambda a:a[ii][:,jj]
    shape=cut(h['sdf']);raw=cut(learned['direct']);aligned=cut(learned['scores'])
    if method.startswith('history_') and method!='history_matched_fusion':return cut(h[method[len('history_'):]])
    if method=='history_matched_fusion':return (centered(cut(h['ridge']))+centered(shape))/2
    if method=='source_ridge':return cut(source['scores'])
    if method=='source_fusion':return (centered(cut(source['scores']))+centered(shape))/2
    if method=='source_direct_fusion':return (centered(cut(source['direct']))+centered(shape))/2
    if method=='R_direct':return raw
    if method=='R_ridge':return aligned
    if method=='R_direct_fusion':return (centered(raw)+centered(shape))/2
    if method=='R_fusion':return (centered(aligned)+centered(shape))/2
    if method=='R_rrf60':return (1/(60+ordinal(aligned))+1/(60+ordinal(shape)))/2
    if method=='R_borda':return -(ordinal(aligned)+ordinal(shape))/2
    raise ValueError(method)
def solve(a,ids,kind):
    ordered=sorted(range(len(ids)),key=lambda j:ids[j]);s=a[:,ordered]
    if kind=='global':rr,cc=linear_sum_assignment(-s)
    elif kind=='independent':rr=np.arange(len(s));cc=np.argmax(s,axis=1)
    else:
        usedr=set();usedc=set();pairs=[]
        for k in np.argsort(-s.ravel(),kind='stable'):
            i,j=divmod(int(k),s.shape[1])
            if i not in usedr and j not in usedc:usedr.add(i);usedc.add(j);pairs.append((i,j))
            if len(pairs)==len(s):break
        rr=np.array([x for x,y in pairs]);cc=np.array([y for x,y in pairs])
    result=[None]*len(s)
    for i,j in zip(rr,cc):result[int(i)]=ids[ordered[int(j)]]
    return result
def acc(rows):
    q={};cad={}
    for r in rows:
        key=r['query_id'];total,count=q.get(key,(0.,0));q[key]=(total+int(r['correct']),count+1);cad[key]=r['true_cad']
    m=defaultdict(list)
    for key,(s,n) in q.items():m[cad[key]].append(s/n)
    return {i:sum(v)/len(v) for i,v in m.items()}
def calc(rows):
    objects=acc(rows);cases=defaultdict(list);batches=defaultdict(set)
    for r in rows:cases[r['case_id']].append(r);batches[r['batch_id']].add(r['case_id'])
    exact={c:all(r['correct'] for r in rs) for c,rs in cases.items()}
    return dict(object_macro_accuracy=sum(objects.values())/len(objects),identities=len(objects),queries=len({r['query_id'] for r in rows}),observations=len(rows),case_count=len(cases),unique_batches=len(batches),
                whole_batch_exact=sum(exact.values())/len(exact),unique_batch_exact_macro=sum(sum(exact[c] for c in cs)/len(cs) for cs in batches.values())/len(batches),
                mean_duplicate_assignments=sum(len(rs)-len({r['prediction'] for r in rs}) for rs in cases.values())/len(cases))
def interval(a,b,qmap):
    aa=acc(a);bb=acc(b);assert set(aa)==set(bb)
    batches=sorted({r['batch_id'] for r in a});members={v:{v} for v in batches};by_group=defaultdict(set)
    for r in a:by_group[qmap[r['query_id']]['group_id']].add(r['batch_id'])
    for group in by_group.values():
        merge=set.union(*(members[v] for v in group))
        for v in merge:members[v]=merge
    model_batch={r['true_cad']:r['batch_id'] for r in a};components={}
    for m in sorted(aa):components.setdefault(tuple(sorted(members[model_batch[m]])),[]).append(m)
    units=sorted(components.values(),key=lambda v:v[0]);d=np.array([sum(aa[m]-bb[m] for m in v) for v in units]);n=np.array([len(v) for v in units])
    rng=np.random.default_rng(20261006);indices=rng.integers(len(units),size=(10000,len(units)))
    samples=d[indices].sum(1)/n[indices].sum(1)
    return dict(difference=float(d.sum()/n.sum()),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),bootstrap_units=len(units),objects=len(aa),replicates=10000)
def main():
    start=time.monotonic();p=read(HERE/'protocol.json');records=p['records'];qm={r['query_id']:r for r in records};qi={q:i for i,q in enumerate(qm)};ci={m:i for i,m in enumerate(p['gallery_model_ids'])}
    definitions={d['batch_id']:d for d in gz(PUB/'inventories/batch_definitions.json.gz')};cases={c['case_id']:c for c in gz(PUB/'inventories/batch_cases.json.gz') if c['scenario']=='closed'}
    with np.load(PUB/'score_cache/scores_438.npz') as f:h={k:f[k].astype('f8') for k in ['cnos_rgb','sdf','ridge','z_fusion','rrf60','borda']}
    with np.load(PUB/'inventories/fusion_scores.npz') as f:h['supervised']=f['scores'].astype('f8')
    with np.load(HERE/'oof_source.npz') as f:source={k:f[k].astype('f8') for k in ['scores','direct']}
    learned={}
    for seed in p['training_seeds']:
        with np.load(HERE/f'oof_R_s{seed}.npz') as f:learned[seed]={k:f[k].astype('f8') for k in ['scores','direct']}
        for fold in range(5):
            with np.load(HERE/f'R_s{seed}_f{fold}/scores.npz') as f:
                ii=[i for i,r in enumerate(records) if r['outer_fold']==fold]
                assert np.array_equal(learned[seed]['scores'][ii],f['scores'][ii])
    scopes=defaultdict(list);mainrows={};cached=None;expected=None;matrix=None;count=0;tie_alternatives=0;objective_errors=[]
    with gzip.open(HERE/'predictions.csv.gz','rt',encoding='utf-8',newline='') as f:
        for raw in csv.DictReader(f):
            r=dict(raw)
            for k in ['training_seed','capacity','layout_seed','fold']:r[k]=int(r[k])
            for k in ['full','correct']:r[k]=r[k]=='True'
            r['assigned_score']=float(r['assigned_score'])
            assert r['true_cad']==qm[r['query_id']]['model_id'] and r['fold']==qm[r['query_id']]['outer_fold']
            assert r['correct']==(r['prediction']==r['true_cad'])
            key=(r['training_seed'],r['case_id'],r['method'],r['strategy'])
            if key!=cached:
                de=definitions[r['batch_id']];case=cases[r['case_id']];qs=case['query_ids'];ids=de['candidate_cad_ids'];ii=[qi[q] for q in qs];jj=[ci[m] for m in ids]
                assert de['fold']==r['fold'] and len(qs)==len(ids)
                matrix=build(r['method'],h,source,learned[r['training_seed']],ii,jj);expected=solve(matrix,ids,r['strategy']);cached=key
            pos=qs.index(r['query_id']);col=ids.index(r['prediction']);assert abs(matrix[pos,col]-r['assigned_score'])<2e-8
            if r['prediction']!=expected[pos]:
                # Equal global optima can use a different tie arrangement. Check the complete objective below.
                if r['strategy']!='global':assert abs(matrix[pos,col]-matrix[pos,ids.index(expected[pos])])<1e-10
                tie_alternatives+=1
            scopes[(r['training_seed'],r['layout'],r['cohort'],r['capacity'],r['layout_seed'],r['method'],r['strategy'])].append(r);count+=1
    largest_error=0.;stats_checked=0
    for published in table(HERE/'all_results.csv'):
        key=(int(published['training_seed']),published['layout'],published['cohort'],int(published['capacity']),int(published['layout_seed']),published['method'],published['strategy'])
        rr=scopes[key];pop=published['population'];rr=[r for r in rr if pop=='all' or r['full']==(pop=='full')]
        actual=calc(rr)
        for k,v in actual.items():largest_error=max(largest_error,abs(float(published[k])-v))
        if published['cohort']=='waxup' and int(published['capacity'])==20 and int(published['layout_seed'])==20261005 and pop=='full':mainrows[(key[0],key[1],key[5],key[6])]=rr
        stats_checked+=1
    assert largest_error<1e-12
    # Verify global objectives and capacities for every unique case/method combination, including ties.
    for key,rr in scopes.items():
        if key[-1]=='independent':continue
        bycase=defaultdict(list)
        for r in rr:bycase[r['case_id']].append(r)
        for cid,rs in bycase.items():
            d=definitions[rs[0]['batch_id']];case=cases[cid];qs=case['query_ids'];ids=d['candidate_cad_ids'];ii=[qi[q] for q in qs];jj=[ci[m] for m in ids]
            assert len(rs)==len(qs) and len({r['prediction'] for r in rs})==len(ids)
            a=build(key[-2],h,source,learned[key[0]],ii,jj)
            if key[-1]=='global':
                x,y=linear_sum_assignment(-a);opt=float(a[x,y].sum());given=sum(r['assigned_score'] for r in rs);objective_errors.append(abs(opt-given))
    assert max(objective_errors,default=0)<1e-7
    ci_error=0.
    for row in table(HERE/'paired_intervals.csv'):
        seed=int(row['training_seed']);a=mainrows[(seed,row['layout'],'R_fusion','global')]
        b=mainrows.get((seed,row['layout'],row['comparator'],row['comparator_strategy']),mainrows.get((20261006,row['layout'],row['comparator'],row['comparator_strategy'])))
        result=interval(a,b,qm)
        for k,v in result.items():ci_error=max(ci_error,abs(float(row[k])-v))
    assert ci_error<1e-10
    absolute_error=0.
    for row in table(HERE/'absolute_intervals.csv'):
        seed=int(row['training_seed']);a=mainrows[(seed,row['layout'],row['method'],row['strategy'])]
        result=interval(a,[dict(r,correct=False) for r in a],qm)
        absolute_error=max(absolute_error,abs(float(row['accuracy'])-result['difference']))
        for k in ['ci_low','ci_high','bootstrap_units','objects','replicates']:absolute_error=max(absolute_error,abs(float(row[k])-result[k]))
    assert absolute_error<1e-10
    # Every retrieval rank is independently obtained by lexicographically sorting raw candidate scores.
    rank_lookup={};rank_error=0
    for row in table(HERE/'retrieval_per_query.csv'):
        seed=int(row['training_seed']);method=row['method'];query=row['query_id'];i=qi[query];truth=ci[qm[query]['model_id']]
        key=(seed,method)
        if key not in rank_lookup:
            if method=='R_fusion':a=(centered(learned[seed]['scores'])+centered(h['sdf']))/2
            elif method=='R_ridge':a=learned[seed]['scores']
            elif method=='R_direct':a=learned[seed]['direct']
            elif method=='source_ridge':a=source['scores']
            elif method=='source_fusion':a=(centered(source['scores'])+centered(h['sdf']))/2
            else:a=h[method[len('history_'):]]
            rank_lookup[key]=ordinal(a)
        rank_error=max(rank_error,abs(int(row['rank'])-int(rank_lookup[key][i,truth])))
    assert rank_error==0
    # Supplementary tables must be traceable to the same predictions too.
    source_kind=read(PUB/'data_groups/protocol.json')['source_kind_by_query']
    common_sets=[]
    for capacity in [10,20,40]:
        common_sets.append({m for d in definitions.values() if d['layout']=='random' and d['cohort']=='waxup' and d['seed']==20261005 and d['capacity']==capacity and d['full_capacity_batch'] for m in d['candidate_cad_ids']})
    common=set.intersection(*common_sets);assert common==set(read(HERE/'shared_capacity_ids.json'))
    supplementary_error=0.;supplementary_rows=0
    for name in ['primary_results.csv','fold_results.csv','capacity_same_ids.csv','source_analysis.csv','development_fold_excluded_results.csv']:
        for row in table(HERE/name):
            seed=int(row['training_seed']);layout=row['layout'];method=row['method'];strategy=row['strategy']
            if name=='development_fold_excluded_results.csv':
                rr=[r for r in mainrows[(seed,layout,method,strategy)] if r['fold']!=int(row['excluded_fold'])]
            else:
                key=(seed,layout,row['cohort'],int(row['capacity']),int(row['layout_seed']),method,strategy)
                rr=scopes[key];pop=row['population']
                if name=='capacity_same_ids.csv':rr=[r for r in rr if r['full'] and r['true_cad'] in common]
                else:rr=[r for r in rr if pop=='all' or r['full']==(pop=='full')]
                if name=='fold_results.csv':rr=[r for r in rr if r['fold']==int(row['fold'])]
                if name=='source_analysis.csv':rr=[r for r in rr if source_kind[r['query_id']]==row['source_kind']]
            actual=calc(rr)
            if name=='source_analysis.csv':actual['groups']=len({r['group_id'] for r in rr})
            for k,v in actual.items():
                if k in row:supplementary_error=max(supplementary_error,abs(float(row[k])-v))
            supplementary_rows+=1
    for row in table(HERE/'layout_sensitivity.csv'):
        selected=[v for key,v in scopes.items() if key[0]==int(row['training_seed']) and key[1]==row['layout'] and key[2]=='waxup' and key[3]==20 and key[5]==row['method'] and key[6]==row['strategy']]
        values=[calc([r for r in rr if r['full']])['object_macro_accuracy'] for rr in selected]
        assert len(values)==int(row['layouts'])==25
        for k,v in [('mean',np.mean(values)),('minimum',min(values)),('maximum',max(values))]:supplementary_error=max(supplementary_error,abs(float(row[k])-v))
        supplementary_rows+=1
    for row in table(HERE/'error_changes.csv'):
        seed=int(row['training_seed']);a=acc(mainrows[(seed,row['layout'],'R_fusion','global')])
        b=acc(mainrows.get((seed,row['layout'],row['comparator'],row['comparator_strategy']),mainrows.get((20261006,row['layout'],row['comparator'],row['comparator_strategy']))))
        actual=dict(improved_ids=sum(a[m]>b[m]+1e-12 for m in a),worsened_ids=sum(a[m]<b[m]-1e-12 for m in a),unchanged_ids=sum(abs(a[m]-b[m])<=1e-12 for m in a),total_gain=sum(max(0,a[m]-b[m]) for m in a),total_loss=sum(max(0,b[m]-a[m]) for m in a))
        for k,v in actual.items():supplementary_error=max(supplementary_error,abs(float(row[k])-v))
        supplementary_rows+=1
    for row in table(HERE/'retrieval_results.csv'):
        key=(int(row['training_seed']),row['method']);indices=[i for i,r in enumerate(records) if int(row['fold'])==-1 or r['outer_fold']==int(row['fold'])]
        by_model=defaultdict(list)
        for i in indices:by_model[records[i]['model_id']].append(rank_lookup[key][i,ci[records[i]['model_id']]])
        actual={('top1' if k==1 else f'recall_at_{k}'):np.mean([np.mean(np.asarray(v)<=k) for v in by_model.values()]) for k in [1,5,10,20,100]}
        actual.update(mrr=np.mean([np.mean(1/np.asarray(v)) for v in by_model.values()]),identities=len(by_model),queries=len(indices))
        for k,v in actual.items():supplementary_error=max(supplementary_error,abs(float(row[k])-v))
        supplementary_rows+=1
    for row in table(HERE/'coverage.csv'):
        covered={r['query_id'] for key,rr in scopes.items() if key[0]==int(row['training_seed']) and key[5]=='R_fusion' and key[6]=='global' for r in rr}
        assert covered==set(qm) and len(covered)==int(row['photos'])==488
        assert len({qm[q]['model_id'] for q in covered})==int(row['identities'])==438
        assert row['all_488_retained']=='True'
    assert supplementary_error<1e-12
    result=dict(status='passed_independent_prediction_assignment_aggregate_interval_rank_verification',prediction_rows=count,
        aggregate_rows_checked=stats_checked,aggregate_max_error=largest_error,paired_interval_max_error=ci_error,
        absolute_interval_max_error=absolute_error,assignment_objectives_checked=len(objective_errors),assignment_objective_max_error=max(objective_errors,default=0),
        tied_alternative_prediction_count=tie_alternatives,retrieval_rank_max_error=rank_error,
        supplementary_rows_checked=supplementary_rows,supplementary_max_error=supplementary_error,
        all_three_seeds_all_488_queries_retained=True,wall_seconds=time.monotonic()-start)
    (HERE/'RESULT_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
