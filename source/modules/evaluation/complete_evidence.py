"""Fixed cached-score post-processing for the approved unmapped manuscript revision.

No encoder inference, fitting, optimization search, or original-file overwrite.
Assignment receives cached scores and inventory membership only. Truth is used
after selection for checks and aggregates. Existing case/identity weights remain.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
from collections import defaultdict
import csv,gzip,json,hashlib,importlib.util,time,sys
import numpy as np

HERE=Path(__file__).resolve().parent
PUB=HERE.parent
ROOT=PUB.parent
OLD=PUB/'pipeline'
assert not (HERE/'STATISTICS_COMPLETE.json').exists(), 'Completed revision is immutable'

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

original=load_module('original_analysis',OLD/'analyze.py')
independent=load_module('independent_verifier',OLD/'verify_results.py')
assign=original.assign
START=time.monotonic()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def readgz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def write(name,rows):
    if not rows:return
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

manifest=read(HERE/'source_manifest.json')
for p in [OLD/'analyze.py',OLD/'verify_results.py']:
    rel=str(p.relative_to(ROOT)).replace('\\','/')
    manifest.setdefault(rel,sha(p))
for rel,digest in manifest.items():assert sha(ROOT/rel)==digest,rel
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
protocol=read(OLD/'protocol.json')
records=protocol['records'];qm={r['query_id']:r for r in records}
qi={r['query_id']:i for i,r in enumerate(records)}
cids=protocol['gallery_model_ids'];ci={m:i for i,m in enumerate(cids)}
prior=read(PUB/'score_cache/protocol.json')
assert [r['query_id'] for r in prior['records']]==list(qm)
assert prior['gallery438']==cids
assert len(records)==488 and len(cids)==438
with np.load(PUB/'score_cache/scores_438.npz') as z:sdf=z['sdf'].astype('f8')
assert sdf.shape==(488,438) and np.isfinite(sdf).all()
definitions={d['batch_id']:d for d in readgz(PUB/'inventories/batch_definitions.json.gz')}
main_partition=20261005
seeds=protocol['training_seeds'];primary=protocol['primary_seed']
assert seeds==[20261006,20261007,20261008] and primary==20261006
cases=[]
for case in readgz(PUB/'inventories/batch_cases.json.gz'):
    d=definitions[case['batch_id']]
    if case['scenario']!='closed' or d['cohort']!='waxup' or not d['full_capacity_batch']:continue
    if d['capacity']==20 or (d['capacity'] in [10,40] and d['layout']=='random' and d['seed']==main_partition):cases.append(case)
assert cases
case_ids={c['case_id'] for c in cases}
old_direct={}
old_main=defaultdict(list)
scanned=0
with gzip.open(OLD/'predictions.csv.gz','rt',encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        scanned+=1
        if r['case_id'] not in case_ids:continue
        seed=int(r['training_seed'])
        if r['method']=='R_direct_fusion' and r['strategy'] in ['independent','global']:
            key=(seed,r['case_id'],r['query_id'],r['strategy'])
            assert key not in old_direct
            old_direct[key]=(r['prediction'],r['correct']=='True')
        if r['cohort']=='waxup' and r['capacity']=='20' and r['layout_seed']==str(main_partition) and r['full']=='True':
            rr=dict(r);rr['correct']=r['correct']=='True';rr['fold']=int(r['fold'])
            old_main[(seed,r['layout'],r['method'],r['strategy'])].append(rr)

summaries=[];folds=[];capacity=[];excluded=[];absolute=[];paired=[];mainrows={}
rebuilt_existing=0;verified_new_solutions=0;maximum_objective_error=0.0
methods=[('R_direct_fusion','independent'),('R_direct_fusion','greedy'),('R_direct_fusion','global'),('R_direct_rrf60','global'),('R_direct_borda','global')]
shared=set(read(OLD/'shared_capacity_ids.json'))
assert len(shared)==160
fields=['training_seed','method','strategy','layout','cohort','capacity','layout_seed','fold','batch_id','case_id','full','query_id','true_cad','prediction','correct','group_id','assigned_score']
SCALAR_KEYS=['object_macro_accuracy','identities','queries','observations','case_count','unique_batches','mean_duplicate_assignments']

def metrics(rows):
    v=original.stats(rows)
    vv=independent.calc(rows)
    for k in SCALAR_KEYS:assert abs(float(v[k])-float(vv[k]))<1e-12,(k,v[k],vv[k])
    return {k:v[k] for k in SCALAR_KEYS}

def verify_interval(a,b):
    result=original.paired(a,b,qm)
    check=independent.interval(a,b,qm)
    for k in ['difference','ci_low','ci_high','bootstrap_units','objects','replicates']:
        assert abs(float(result[k])-float(check[k]))<1e-12,(k,result,check)
    assert result['bootstrap_units']==12 and result['objects']==240 and result['replicates']==10000
    return result

with gzip.open(HERE/'predictions_unmapped.csv.gz','wt',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for seed in seeds:
        direct=np.zeros((488,438),dtype='f8');covered=np.zeros(488,int)
        for fold in range(5):
            with np.load(OLD/f'R_s{seed}_f{fold}/scores.npz') as z:
                assert z['query_ids'].tolist()==list(qm) and z['gallery_ids'].tolist()==cids
                ii=np.array([i for i,r in enumerate(records) if r['outer_fold']==fold])
                direct[ii]=z['direct'][ii];covered[ii]+=1
        assert np.all(covered==1) and np.isfinite(direct).all()
        scopes=defaultdict(list)
        for case in cases:
            d=definitions[case['batch_id']];qs=case['query_ids'];candidate_ids=d['candidate_cad_ids']
            ii=[qi[q] for q in qs];jj=[ci[m] for m in candidate_ids]
            assert len(qs)==len(candidate_ids)==d['capacity'] and all(n==1 for n in d['counts'])
            assert len(set(qs))==len(qs)
            assert all(qm[q]['outer_fold']==d['fold'] for q in qs)
            appearance=direct[ii][:,jj];shape=sdf[ii][:,jj]
            ar=original.rank_matrix(appearance);sr=original.rank_matrix(shape)
            assert np.array_equal(ar,independent.ordinal(appearance)) and np.array_equal(sr,independent.ordinal(shape))
            matrices={'R_direct_fusion':.5*(original.z(appearance)+original.z(shape)),
                      'R_direct_rrf60':.5*(1/(60+ar)+1/(60+sr)),
                      'R_direct_borda':-.5*(ar+sr)}
            alternate_fusion=(independent.centered(appearance)+independent.centered(shape))/2
            assert np.max(abs(matrices['R_direct_fusion']-alternate_fusion))<1e-12
            for method,strategy in methods:
                matrix=matrices[method]
                selection=assign(matrix,candidate_ids,d['counts'],strategy)
                alt=independent.solve(matrix,candidate_ids,strategy)
                assert selection['predictions']==alt,(seed,case['case_id'],method,strategy)
                if strategy!='independent':assert len(set(selection['predictions']))==len(qs)
                if method=='R_direct_fusion' and strategy in ['independent','global']:
                    for q,y in zip(qs,selection['predictions']):
                        existing=old_direct[(seed,case['case_id'],q,strategy)]
                        assert y==existing[0] and (y==qm[q]['model_id'])==existing[1]
                        rebuilt_existing+=1
                else: verified_new_solutions+=1
                cols={m:j for j,m in enumerate(candidate_ids)}
                selected_scores=[float(matrix[i,cols[y]]) for i,y in enumerate(selection['predictions'])]
                maximum_objective_error=max(maximum_objective_error,abs(sum(selected_scores)-selection['objective']))
                for pos,(q,y) in enumerate(zip(qs,selection['predictions'])):
                    rr=dict(training_seed=seed,method=method,strategy=strategy,layout=d['layout'],cohort=d['cohort'],capacity=d['capacity'],layout_seed=d['seed'],fold=d['fold'],batch_id=d['batch_id'],case_id=case['case_id'],full=True,query_id=q,true_cad=qm[q]['model_id'],prediction=y,correct=y==qm[q]['model_id'],group_id=qm[q]['group_id'],assigned_score=selected_scores[pos])
                    writer.writerow(rr)
                    meta=(seed,d['layout'],d['cohort'],d['capacity'],d['seed'],method,strategy)
                    scopes[meta].append(rr)
        for meta,rows in scopes.items():
            seed1,layout,cohort,cap,partition,method,strategy=meta
            info=dict(training_seed=seed1,layout=layout,cohort=cohort,capacity=cap,layout_seed=partition,method=method,strategy=strategy,population='full')
            summaries.append(dict(**info,**metrics(rows)))
            for fold in sorted({r['fold'] for r in rows}):folds.append(dict(**info,fold=fold,**metrics([r for r in rows if r['fold']==fold])))
            if cap==20 and partition==main_partition:
                mainrows[(seed1,layout,method,strategy)]=rows
                sub=[r for r in rows if r['fold']!=3]
                m=metrics(sub);assert m['identities']==180
                excluded.append(dict(training_seed=seed1,layout=layout,method=method,strategy=strategy,excluded_fold=3,interpretation='retrospective sensitivity, not an independent test',**m))
            if layout=='random' and partition==main_partition:
                sub=[r for r in rows if r['true_cad'] in shared]
                m=metrics(sub);assert m['identities']==160 and m['queries']==164
                capacity.append(dict(**{k:v for k,v in info.items() if k!='population'},population='same_ids_within_original_full_batches',**{k:m[k] for k in ['object_macro_accuracy','identities','queries','observations']}))
        print(json.dumps({'statistics_seed_complete':seed,'existing_prediction_rows_verified':rebuilt_existing,'elapsed_seconds':round(time.monotonic()-START,1)}),flush=True)

assert rebuilt_existing==len(old_direct)
expected_all=readcsv(OLD/'all_results.csv');old_abs=readcsv(OLD/'absolute_intervals.csv')
old_cap=readcsv(OLD/'capacity_same_ids.csv');old_excluded=readcsv(OLD/'development_fold_excluded_results.csv');old_pair=readcsv(OLD/'paired_intervals.csv')
def row_key(r,keys):return tuple(str(r[k]) for k in keys)
allkeys=['training_seed','layout','cohort','capacity','layout_seed','method','strategy','population']
lookup={row_key(r,allkeys):r for r in expected_all}
matched_summaries=0
for row in summaries:
    if row['method']!='R_direct_fusion' or row['strategy']=='greedy':continue
    expected=lookup[row_key(row,allkeys)]
    for k in SCALAR_KEYS:assert abs(float(row[k])-float(expected[k]))<1e-12,(row,k)
    matched_summaries+=1

capkeys=['training_seed','layout','cohort','capacity','layout_seed','method','strategy','population']
caplookup={row_key(r,capkeys):r for r in old_cap};matched_capacity=0
for row in capacity:
    if row['method']=='R_direct_fusion' and row['strategy'] in ['independent','global']:
        old=caplookup[row_key(row,capkeys)]
        for k in ['object_macro_accuracy','identities','queries','observations']:assert abs(float(row[k])-float(old[k]))<1e-12
        matched_capacity+=1
devkeys=['training_seed','layout','method','strategy','excluded_fold']
devlookup={row_key(r,devkeys):r for r in old_excluded};matched_excluded=0
for row in excluded:
    if row['method']=='R_direct_fusion' and row['strategy'] in ['independent','global']:
        old=devlookup[row_key(row,devkeys)]
        for k in SCALAR_KEYS:assert abs(float(row[k])-float(old[k]))<1e-12
        matched_excluded+=1

abslookup={row_key(r,['training_seed','layout','method','strategy']):r for r in old_abs}
absolute_reused=0
for (seed,layout,method,strategy),rows in mainrows.items():
    zero=[dict(r,correct=False) for r in rows]
    result=verify_interval(rows,zero)
    key=(str(seed),layout,method,strategy)
    origin='computed from fixed cached scores'
    if key in abslookup:
        existing=abslookup[key]
        for k,a in [('accuracy','difference'),('ci_low','ci_low'),('ci_high','ci_high')]:assert abs(float(existing[k])-result[a])<1e-12
        origin='existing absolute interval verified and reused';absolute_reused+=1
        result['difference']=float(existing['accuracy']);result['ci_low']=float(existing['ci_low']);result['ci_high']=float(existing['ci_high'])
    absolute.append(dict(training_seed=seed,layout=layout,method=method,strategy=strategy,accuracy=result['difference'],ci_low=result['ci_low'],ci_high=result['ci_high'],bootstrap_units=12,objects=240,replicates=10000,conditional=result['conditional'],origin=origin))

comparators=[('R_direct_fusion','independent'),('R_direct_fusion','greedy'),('history_z_fusion','global'),('history_matched_fusion','global'),('history_supervised','global'),('source_fusion','global'),('source_direct_fusion','global'),('R_fusion','global'),('R_direct','global'),('R_ridge','global'),('history_sdf','global'),('R_direct_rrf60','global'),('R_direct_borda','global'),('history_cnos_rgb','global')]
pair_reused=0
for seed in seeds:
    for layout in ['random','hard']:
        a=mainrows[(seed,layout,'R_direct_fusion','global')]
        for method,strategy in comparators:
            b=mainrows.get((seed,layout,method,strategy))
            comparator_seed=seed
            if b is None:b=old_main.get((seed,layout,method,strategy))
            if b is None:b=old_main.get((primary,layout,method,strategy));comparator_seed=primary
            assert b is not None,(seed,layout,method,strategy)
            result=verify_interval(a,b)
            assert {(r['case_id'],r['query_id']) for r in a}=={(r['case_id'],r['query_id']) for r in b}
            origin='computed paired connected-unit interval'
            if method=='R_fusion' and strategy=='global':
                candidates=[r for r in old_pair if r['training_seed']==str(seed) and r['layout']==layout and r['method']=='R_fusion' and r['strategy']=='global' and r['comparator']=='R_direct_fusion' and r['comparator_strategy']=='global']
                assert len(candidates)==1
                old=candidates[0];reverse={'difference':-float(old['difference']),'ci_low':-float(old['ci_high']),'ci_high':-float(old['ci_low'])}
                for k,v in reverse.items():assert abs(result[k]-v)<1e-12
                result.update(reverse);origin='existing reverse paired interval verified and reused';pair_reused+=1
            paired.append(dict(training_seed=seed,layout=layout,method='R_direct_fusion',strategy='global',comparator=method,comparator_strategy=strategy,comparator_training_seed=comparator_seed,**result,origin=origin))

sensitivity=[];layout_pairs=[]
for seed in seeds:
    for layout in ['random','hard']:
        for method,strategy in methods:
            rows=[r for r in summaries if r['training_seed']==seed and r['layout']==layout and r['capacity']==20 and r['method']==method and r['strategy']==strategy]
            assert len(rows)==25
            values=[r['object_macro_accuracy'] for r in rows]
            sensitivity.append(dict(training_seed=seed,layout=layout,method=method,strategy=strategy,mean=float(np.mean(values)),minimum=min(values),maximum=max(values),layouts=25,independent_samples=False))
        for partition in sorted({r['layout_seed'] for r in summaries if r['capacity']==20 and r['layout']==layout}):
            rows=[r for r in summaries if r['training_seed']==seed and r['layout']==layout and r['capacity']==20 and r['layout_seed']==partition and r['method']=='R_direct_fusion']
            values={r['strategy']:r['object_macro_accuracy'] for r in rows};assert set(values)=={'independent','greedy','global'}
            layout_pairs.append(dict(training_seed=seed,layout=layout,layout_seed=partition,**values,global_minus_independent=values['global']-values['independent'],global_minus_greedy=values['global']-values['greedy'],main_partition=partition==main_partition,independent_samples=False))

primary_rows=[r for r in summaries if r['training_seed']==primary and r['capacity']==20 and r['layout_seed']==main_partition]
comparison_methods=['history_cnos_rgb','history_sdf','history_z_fusion','history_matched_fusion','history_supervised','source_fusion','R_fusion','R_direct_fusion']
main_comparison=[]
for layout in ['random','hard']:
    for method in comparison_methods:
        rows=mainrows.get((primary,layout,method,'global'),old_main.get((primary,layout,method,'global')))
        assert rows
        point=metrics(rows)
        cirow=next(r for r in (absolute if method=='R_direct_fusion' else old_abs) if str(r['training_seed'])==str(primary) and r['layout']==layout and r['method']==method and r['strategy']=='global')
        main_comparison.append(dict(condition=layout,method=method,**point,ci_low=float(cirow['ci_low']),ci_high=float(cirow['ci_high']),main_pipeline=method=='R_direct_fusion'))

for name,rows in [('all_results_unmapped.csv',summaries),('primary_results_unmapped.csv',primary_rows),('fold_results_unmapped.csv',folds),('absolute_intervals_unmapped.csv',absolute),('paired_intervals_unmapped.csv',paired),('layout_sensitivity_unmapped.csv',sensitivity),('layout_pairs_unmapped.csv',layout_pairs),('capacity_same_ids_unmapped.csv',capacity),('development_fold_excluded_unmapped.csv',excluded),('main_comparison_unmapped.csv',main_comparison)]:write(name,rows)

for rel,digest in manifest.items():assert sha(ROOT/rel)==digest,rel
summary={'status':'statistics_complete_self_checked; timing and user acceptance pending','new_training':False,'new_feature_extraction':False,'parameter_search':False,'main_method':'R_direct_fusion','seed_selection_unchanged':True,'population_and_arrangements_unchanged':True,'original_prediction_rows_scanned':scanned,'existing_prediction_rows_verified_exactly':rebuilt_existing,'existing_summaries_verified':matched_summaries,'existing_absolute_intervals_verified_reused':absolute_reused,'reverse_paired_intervals_verified_reused':pair_reused,'existing_capacity_rows_verified':matched_capacity,'existing_development_excluded_rows_verified':matched_excluded,'new_assignment_solutions_independently_verified':verified_new_solutions,'maximum_assignment_objective_reconstruction_error':maximum_objective_error,'summary_rows':len(summaries),'primary_rows':len(primary_rows),'absolute_interval_rows':len(absolute),'paired_interval_rows':len(paired),'layout_pair_rows':len(layout_pairs),'capacity_rows':len(capacity),'fold_rows':len(folds),'selected_case_count':len(cases),'protected_inputs':len(manifest),'protected_manuscript_files':len(paper_before),'main_results':primary_rows,'elapsed_seconds':time.monotonic()-START,'user_acceptance':'pending','next_unit_started':False}
(HERE/'STATISTICS_COMPLETE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='main_results'},ensure_ascii=False,indent=2))
