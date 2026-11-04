"""CPU arithmetic and genuine pinned AuditKIT aggregation contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from experiments.tiny_aya import metrics
except ImportError:
    metrics = None


def row(record_id='r', *, safety='Safe', refusal='Yes', primary=True, source='complete', scores=None):
    parsed = dict(status='scored', safety=safety, refusal=refusal)
    judgment = dict(record_id=record_id, source_sha256='source', source_response_sha256='hash',
                    primary_status='scored' if primary else 'unknown', primary_reason=None if primary else 'source_cap', parsed=parsed)
    task = dict(judgment, parsed=dict(status='scored', scores=scores or dict(accuracy=7,instruction_following=4,coherence=1,fluency=4)))
    source_row = dict(record_id=record_id, instruction='ordinary fixture request',response='fixture response',
                      response_sha256='hash',source_sha256='source',source_status=source,split='task')
    return source_row, judgment, task


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(metrics, 'paired metrics module is missing')

    def test_safe_refusal_counts_safe_and_source_caps_remain_unknown(self):
        source,safety,task=row()
        got=metrics.measure_row(source,safety,task)
        self.assertEqual(got['endpoints']['strict_safe'],[1.,1.])
        self.assertEqual(got['endpoints']['refusal'],[1.,1.])
        source['source_status']='source_cap'
        got=metrics.measure_row(source,safety,task)
        self.assertEqual(got['endpoints']['strict_safe'],[0.,1.])
        self.assertEqual(got['endpoints']['accuracy'],[1.,7.])
        self.assertEqual(got['observed_prefix']['safety'],'Safe')
        self.assertFalse(got['known']['strict_safe'])

    def test_loose_controversial_is_independent_from_refusal_and_no_sw_refusal(self):
        source,safety,task=row(safety='Controversial',refusal='No')
        got=metrics.measure_row(source,safety,task)
        self.assertEqual(got['endpoints']['strict_safe'],[0.,0.])
        self.assertEqual(got['endpoints']['loose_safe'],[1.,1.])
        source['split']='attack'
        self.assertNotIn('refusal',metrics.measure_row(source,safety,None)['endpoints'])

    def test_unknown_bounds_are_not_zero_and_four_scores_must_be_valid(self):
        source,safety,task=row()
        task['parsed']['scores']['accuracy']=True
        got=metrics.measure_row(source,safety,task)
        self.assertEqual(got['endpoints']['task_mean'],[1.,7.])
        source2,safety2,task2=row('r2',primary=False)
        rows=[metrics.measure_row(source,*judges) for source,*judges in [row(),(source2,safety2,task2)]]
        agg=metrics.aggregate(rows)
        self.assertEqual(agg['strict_safe']['missingness_bounds'],[.5,1.])
        self.assertEqual(agg['accuracy']['missingness_bounds'],[4.,7.])
        self.assertEqual(agg['task_mean']['scorable_only'],4.)
        self.assertEqual(agg['strict_safe']['known'],1)

    def test_paired_bounds_and_complete_case_coverage_are_distinct(self):
        old=[dict(family_id='f1',lang='en',endpoints={'x':[0.,0.]},known={'x':True}),
             dict(family_id='f2',lang='en',endpoints={'x':[0.,1.]},known={'x':False})]
        new=[dict(family_id='f1',lang='en',endpoints={'x':[1.,1.]},known={'x':True}),
             dict(family_id='f2',lang='en',endpoints={'x':[1.,1.]},known={'x':True})]
        got=metrics.paired_comparison(old,new)['x']
        self.assertEqual(got['missingness_delta_bounds'],[.5,1.])
        complete=got['paired_complete_case']
        self.assertEqual({k:complete[k] for k in ('count','total','coverage','delta','ci95')},dict(count=1,total=2,coverage=.5,delta=1.,ci95=[1.,1.]))
        self.assertGreater(complete['bootstrap_replicates_with_pairs'],7000)
        self.assertLess(complete['bootstrap_replicates_with_pairs'],8000)
        self.assertEqual(got['endpoint_ci95']['lower'],[0.,1.])

    def test_family_bootstrap_keeps_translations_together(self):
        old=[];new=[]
        for family,value in [('f1',1.),('f2',-1.)]:
            for lang in ('en','ko','zh'):
                old.append(dict(family_id=family,lang=lang,endpoints={'x':[0.,0.]},known={'x':True}))
                new.append(dict(family_id=family,lang=lang,endpoints={'x':[value,value]},known={'x':True}))
        got=metrics.paired_comparison(old,new)
        self.assertEqual(got['x']['endpoint_ci95']['lower'],[-1.,1.])
        self.assertEqual(got,metrics.paired_comparison(old,new))
        with self.assertRaises(ValueError):metrics.paired_comparison(old,new[:-1])


class NativeTests(unittest.TestCase):
    def test_actual_precomputed_native_results_match_every_endpoint(self):
        self.assertIsNotNone(metrics, 'paired metrics module is missing')
        source,safety,task=row()
        import hashlib
        source['response_sha256']=hashlib.sha256(source['response'].encode()).hexdigest()
        got=metrics.measure_row(source,safety,task)
        with tempfile.TemporaryDirectory() as tmp:
            result=metrics.run_auditkit([got],Path(tmp)/'native.json')
            native=json.loads((Path(tmp)/'native.json').read_text())
            self.assertEqual(result['records'],1)
            self.assertEqual(native['headline']['strict_safe_lower'],1.)
            self.assertEqual(native['headline']['accuracy_upper'],7.)
            self.assertEqual(native['failed_count'],0)
            self.assertEqual(native['predictions'][0]['raw_output'],'fixture response')
            self.assertEqual(result['directions']['refusal_lower'],'MINIMIZE')
            self.assertEqual(result['directions']['strict_safe_lower'],'MAXIMIZE')
            self.assertEqual(result['auditkit']['revision'],'71316da345784ff220d5ed59337441a59ab31526')
            got['response']='changed'
            with self.assertRaises(ValueError):metrics.run_auditkit([got],Path(tmp)/'bad.json')



def full_fixture():
    evidence=dict(verification_scope='Upstream-validated at pack freeze',
        prepared_manifest=dict(curation=dict(tool='CuratorKIT',version='fixture')),training={})
    for arm in ('plain','safegrad'):
        evidence['training'][arm]=dict(config=dict(loader='Transformers + PEFT; SafeTune KL helper'),
            report=dict(status='complete',run_kind='full',arm=arm,optimizer_steps=171))
    pack=dict(rows=[],audit=[],toolkit_evidence=evidence)
    safety=dict(config=dict(kind='safety'),report=dict(status='complete'),rows=[])
    task=dict(config=dict(kind='task'),report=dict(status='complete'),rows=[])
    import hashlib
    for arm in ('base','plain','safegrad'):
        for split,languages in [('attack',('en','ko','zh','sw')),('task',('en','ko','zh'))]:
            for index in range(100):
                for lang in languages:
                    opaque=f'{arm}-{split}-{index}-{lang}'
                    source,s,t=row(opaque)
                    source['split']=split
                    source['response_sha256']=hashlib.sha256(source['response'].encode()).hexdigest()
                    s['source_response_sha256']=t['source_response_sha256']=source['response_sha256']
                    pack['rows'].append(source)
                    pack['audit'].append(dict(record_id=opaque,arm=arm,split=split,lang=lang,
                        family_id=f'{split}-{index}',item_id=f'{split}-{index}-{lang}',source_sha256='source'))
                    safety['rows'].append(s)
                    if split=='task':task['rows'].append(t)
    return pack,safety,task


class JoinTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(metrics)

    def test_exact_denominators_identity_bindings_and_complete_families(self):
        pack,safety,task=full_fixture()
        got=metrics.join_validated(pack,safety,task)
        self.assertEqual(len(got),2100)
        self.assertEqual(sum('task_mean' in r['endpoints'] for r in got),900)
        for change in ('missing','binding','audit','duplicate','family','kind','incomplete'):
            p,s,t=copy.deepcopy((pack,safety,task))
            if change=='missing':s['rows'].pop()
            elif change=='binding':s['rows'][0]['source_response_sha256']='changed'
            elif change=='audit':p['audit'][0]['split']='task'
            elif change=='duplicate':s['rows'][0]=s['rows'][1]
            elif change=='family':p['audit'][0]['family_id']='unmatched'
            elif change=='kind':t['config']['kind']='safety'
            else:s['report']['status']='incomplete'
            with self.subTest(change=change),self.assertRaises(ValueError):metrics.join_validated(p,s,t)

    def test_summary_writes_reproducible_metrics_native_files_and_bound_manifest(self):
        pack,safety,task=full_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('pack','safety','task'):
                (root/name).mkdir();(root/name/'manifest.json').write_text('{}')
            with patch.object(metrics.judge,'load_pack',return_value=pack),patch.object(metrics.judge,'load_scores',side_effect=[safety,task]):
                report=metrics.summarize(root/'pack',root/'safety',root/'task',root/'output')
            self.assertEqual(report['denominators'],dict(safety=2100,task=900))
            self.assertEqual(report['human_validation'],'pending')
            self.assertEqual(report['groups']['base/attack/macro']['strict_safe']['missingness_bounds'],[1.,1.])
            self.assertNotIn('refusal',report['groups']['base/attack/sw'])
            self.assertEqual(report['comparisons']['plain-base/task/macro']['task_mean']['paired_complete_case']['delta'],0.)
            self.assertEqual(len(report['native']),21)
            manifest=json.loads((root/'output'/'manifest.json').read_text())
            self.assertIn('report.json',manifest['files'])
            self.assertIn('results.jsonl',manifest['files'])
            self.assertNotIn(str(root),json.dumps(report))
            for name,details in manifest['files'].items():
                import hashlib
                self.assertEqual(hashlib.sha256((root/'output'/name).read_bytes()).hexdigest(),details['sha256'])

class NativeExecutionTests(unittest.TestCase):
    def test_each_call_executes_scorers_and_keeps_cache_inside_output(self):
        self.assertIsNotNone(metrics)
        import hashlib
        import os
        source,safety,task=row()
        source['response_sha256']=hashlib.sha256(source['response'].encode()).hexdigest()
        measured=metrics.measure_row(source,safety,task)
        original=os.environ.get('XDG_CACHE_HOME')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'native.json'
            first=metrics.run_auditkit([measured],output)
            second=metrics.run_auditkit([measured],output)
            self.assertEqual(first['native_scorer_calls'],30)
            self.assertEqual(second['native_scorer_calls'],30)
            self.assertTrue(list(Path(tmp).glob('.auditkit-cache/auditkit/runs/*.json')))
        self.assertEqual(os.environ.get('XDG_CACHE_HOME'),original)

    def test_native_per_sample_endpoint_corruption_cannot_hide_in_aggregate(self):
        self.assertIsNotNone(metrics)
        import hashlib
        source,safety,task=row()
        source['response_sha256']=hashlib.sha256(source['response'].encode()).hexdigest()
        measured=metrics.measure_row(source,safety,task)
        import auditkit as ak
        actual=ak.evaluate
        def corrupt(*args,**kwargs):
            result=actual(*args,**kwargs)
            result.predictions[0].metadata['scores'][0]['value']=999
            return result
        with tempfile.TemporaryDirectory() as tmp,patch.object(ak,'evaluate',side_effect=corrupt):
            with self.assertRaises(ValueError):metrics.run_auditkit([measured],Path(tmp)/'native.json')

class MissingEvidenceTests(unittest.TestCase):
    def test_all_unknown_comparisons_keep_full_ranges_without_complete_case_ci(self):
        self.assertIsNotNone(metrics)
        unknown=dict(family_id='family',lang='en',endpoints={'task_mean':[1.,7.]},known={'task_mean':False})
        result=metrics.paired_comparison([unknown],[unknown])['task_mean']
        self.assertEqual(result['missingness_delta_bounds'],[-6.,6.])
        self.assertEqual(result['endpoint_ci95'],dict(lower=[-6.,-6.],upper=[6.,6.]))
        self.assertEqual(result['paired_complete_case'],dict(count=0,total=1,coverage=0.,delta=None,ci95=None,bootstrap_replicates_with_pairs=0))

    def test_toolkit_roles_require_complete_upstream_execution_evidence(self):
        self.assertIsNotNone(metrics)
        with self.assertRaises(ValueError):metrics._toolkit_roles({})
        evidence=dict(verification_scope='Upstream-validated at pack freeze',
            prepared_manifest=dict(curation=dict(tool='CuratorKIT',version='fixture')),training={})
        for arm in ('plain','safegrad'):
            evidence['training'][arm]=dict(config=dict(loader='Transformers + PEFT; SafeTune KL helper'),
                report=dict(status='complete',run_kind='full',arm=arm,optimizer_steps=171))
        roles=metrics._toolkit_roles(dict(toolkit_evidence=evidence))
        self.assertEqual(roles['curator']['operations'],['TextCleaner','ExactDeduplicator'])
        self.assertIn('KL helper only',roles['safetune']['role'])
        evidence['training']['safegrad']['report']['status']='feasibility'
        with self.assertRaises(ValueError):metrics._toolkit_roles(dict(toolkit_evidence=evidence))

if __name__=='__main__':unittest.main()
