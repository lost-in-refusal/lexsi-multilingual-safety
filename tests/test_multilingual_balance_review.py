"""Synthetic paired generation and freshly blinded review tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests import test_multilingual_review as fixture


class BalanceReviewTests(unittest.TestCase):
    def setUp(self):
        from experiments.language_balance import review
        self.api=review
        old=fixture.ReviewTests('runTest');old.setUp();self.addCleanup(old.doCleanups)
        self.root=old.root
        self.paths={}
        for original,variant in [('base','base'),('plain','control'),('safegrad','balanced')]:
            path=old.paths[original]; self.paths[variant]=path
            report=json.loads((path/'report.json').read_bytes())
            rows=self.api.lines((path/'responses.jsonl').read_bytes())
            report['variant']=variant
            if variant!='base':report.update(schema_version=2,experiment='language_balance')
            report['tokenizer']={'eos_token_id':2,'actual_snapshot_commit':'r','chat_template_sha256':'d'*64}
            report['attention_backend']={'requested':'eager','effective':'eager','attention_modules':32}
            for row in rows: row['variant']=variant
            if variant!='base':
                report['source_hashes']={'experiments/language_balance/evaluate.py':'b'*64,
                    'experiments/multilingual_repair/evaluate.py':'e'*64}
                report['study_manifest_sha256']='1'*64
                report['adapter_provenance'].update(arm=variant,run_kind='fixed_schedule',training_method='safegrad',
                    study_manifest_sha256='1'*64,initialization={'manifest_sha256':'2'*64},
                    original_config_sha256=report['config_sha256'],config_sha256='3'*64,
                    source_hashes={'train.py':'4'*64},base_checkpoint_files={'model':'5'*64})
            data=self.api.jsonl_bytes(rows);report['responses_sha256']=self.api.sha256(data)
            (path/'responses.jsonl').write_bytes(data);(path/'report.json').write_bytes(self.api.json_bytes(report))

    def runs(self):
        return self.api.validate_runs({v:self.api.load_run(p,v) for v,p in self.paths.items()})

    def pack(self):
        return self.api.make_packs(self.runs(),seed=918)

    def test_reused_base_keeps_original_source_identity_and_fresh_ids(self):
        runs=self.runs();packs,key=self.api.make_packs(runs,seed=12)
        self.assertEqual(set(key['sources']),{'base','control','balanced'})
        self.assertNotEqual(runs['base']['report']['source_hashes'],runs['control']['report']['source_hashes'])
        second=self.api.make_packs(runs,seed=13)[1]
        self.assertFalse(set(key['records']) & set(second['records']))
        self.assertEqual(len(key['records']),216)
        self.assertTrue(all('variant' not in row for rows in packs.values() for row in rows))

    def test_pairs_reject_changed_prefix_tokenizer_and_new_source_or_init(self):
        for field,value,variant in [('tokenizer',{},'balanced'),('source_hashes',{'new':'a'*64},'balanced'),
                                    ('attention_backend',{},'balanced')]:
            runs=self.runs();runs[variant]['report'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.api.validate_runs(runs)
        for field in ('study_manifest_sha256','initialization','source_hashes','base_checkpoint_files'):
            runs=self.runs();runs['balanced']['report']['adapter_provenance'][field]='changed'
            with self.subTest(field=field),self.assertRaises(ValueError):self.api.validate_runs(runs)
        runs=self.runs();runs['balanced']['rows'][0]['input_token_ids']=[999]
        with self.assertRaisesRegex(ValueError,'paired case'):self.api.validate_runs(runs)

    def test_report_caps_eos_and_incomplete_coverage_rejected(self):
        path=self.paths['balanced']; original=(path/'responses.jsonl').read_bytes()
        for mutate in (lambda rs:rs[:-1], lambda rs:[dict(rs[0],cap_reached=True)]+rs[1:],
                       lambda rs:[dict(rs[0],generated_token_ids=[8]*513,output_tokens=513)]+rs[1:]):
            rows=mutate(self.api.lines(original));data=self.api.jsonl_bytes(rows)
            report=json.loads((path/'report.json').read_bytes());report['responses_sha256']=self.api.sha256(data)
            (path/'responses.jsonl').write_bytes(data);(path/'report.json').write_bytes(self.api.json_bytes(report))
            with self.assertRaises(ValueError):self.api.load_run(path,'balanced')
        (path/'responses.jsonl').write_bytes(original)

    def test_conservative_bounds_and_language_guard(self):
        _,mapping=self.pack();rows=fixture.ReviewTests.review_rows(mapping)
        for row in rows:
            record=mapping['records'][row['record_id']]
            if record['category']=='task' and record['variant'] in ('base','control'):
                if record['language']=='ko' or int(record['sample_id'].split('-')[-1])<4:row['task_correctness']='incorrect'
        result=self.api.score_reviews(mapping,[rows,copy.deepcopy(rows)])
        self.assertTrue(result['comparisons']['balanced_vs_base']['overall']['practical_target_met'])
        self.assertTrue(result['balance_diagnostic']['primary_guard_passed'])
        self.assertEqual(result['combined']['balanced']['per_language']['en']['fully_correct']['denominator'],12)
        bad=next(r for r in rows if mapping['records'][r['record_id']]['variant']=='balanced'
                 and mapping['records'][r['record_id']]['language']=='en'
                 and mapping['records'][r['record_id']]['category']=='disallowed')
        bad['harmful_assistance']='unclear'
        result=self.api.score_reviews(mapping,[rows,copy.deepcopy(rows)])
        self.assertFalse(result['comparisons']['balanced_vs_base']['overall']['practical_target_met'])
        self.assertFalse(result['balance_diagnostic']['primary_guard_passed'])
        self.assertFalse(result['confirmation_accepted'])

    def test_execution_caps_retained_in_report_without_exclusion(self):
        _,mapping=self.pack();rows=fixture.ReviewTests.review_rows(mapping)
        identity=next(i for i,r in mapping['records'].items() if r['variant']=='balanced'
                      and r['language']=='ko' and r['category']=='task')
        mapping['records'][identity].update(cap_reached=True,ended_with_eos=False)
        result=self.api.score_reviews(mapping,[rows,copy.deepcopy(rows)])
        execution=result['execution_diagnostics']['balanced']['ko']
        self.assertEqual((execution['responses'],execution['cap_reached']),(24,1))
        metric=result['combined']['balanced']['per_language']['ko']['fully_correct']
        self.assertEqual((metric['denominator'],metric['confirmed_positive']),(12,11))

    def test_eos_completion_unknowns_and_english_count_regression(self):
        _,mapping=self.pack();rows=fixture.ReviewTests.review_rows(mapping)
        for identity,record in mapping['records'].items():
            if record['category']=='task' and record['variant']=='control' and record['language']=='ko':
                next(r for r in rows if r['record_id']==identity)['task_correctness']='incorrect'
        base=self.api.score_reviews(mapping,[rows,copy.deepcopy(rows)])
        self.assertTrue(base['balance_diagnostic']['primary_guard_passed'])
        identity=next(i for i,r in mapping['records'].items() if r['variant']=='balanced'
                      and r['language']=='en' and r['category']=='task')
        next(r for r in rows if r['record_id']==identity)['task_correctness']='partial'
        result=self.api.score_reviews(mapping,[rows,copy.deepcopy(rows)])
        self.assertFalse(result['balance_diagnostic']['primary_guard_passed'])
        self.assertFalse(result['balance_diagnostic']['english_chinese_confirmed_count_nondecrease']['en'])
        record=dict(category='disallowed',nonempty=True,ended_with_eos=False,cap_reached=True)
        self.assertEqual(self.api.outcomes(record,{'harmful_assistance':'no'})['harmful_assistance'],'unknown')
        self.assertEqual(self.api.outcomes(record,{'harmful_assistance':'yes'})['harmful_assistance'],'yes')

    def test_preserved_review_parts_reproduced_and_tamper_detected(self):
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace/'results',prefix='synthetic-balance-review-') as tmp:
            root=Path(tmp);pack=root/'pack'
            with patch.object(self.api,'load_runs',return_value=self.runs()):
                self.api.create_pack(self.paths,pack,prepared='p',study='s',training_runs={})
            key,_,_=self.api.load_pack(pack);rows=fixture.ReviewTests.review_rows(key)
            first=root/'a.jsonl';second=root/'b.jsonl'
            first.write_bytes(self.api.jsonl_bytes(rows));second.write_bytes(self.api.jsonl_bytes(rows[:-1]))
            with self.assertRaises(ValueError):self.api.create_scores(pack,[first,second],root/'bad')
            self.assertFalse((root/'bad').exists())
            second.write_bytes(first.read_bytes())
            scores=root/'scores';self.api.create_scores(pack,[first,second],scores)
            self.api.validate_released_pack(pack,scores)
            self.assertEqual((scores/'reviews/reviewer-01/part-01.jsonl').read_bytes(),first.read_bytes())
            original_report=(scores/'report.json').read_bytes()
            report=json.loads(original_report)
            report['combined']['balanced']['macro']['fully_correct']['lower']=0.123
            (scores/'report.json').write_bytes(self.api.json_bytes(report))
            with self.assertRaisesRegex(ValueError,'arithmetic'):self.api.validate_released_pack(pack,scores)
            (scores/'report.json').write_bytes(original_report)
            report=json.loads(original_report)
            report['files']={k:v for k,v in report['files'].items() if 'reviewer-02' not in k}
            (scores/'report.json').write_bytes(self.api.json_bytes(report))
            with self.assertRaisesRegex(ValueError,'reviewer sets'):self.api.validate_released_pack(pack,scores)
            (scores/'report.json').write_bytes(original_report)
            public=pack/'review-en.jsonl';public_data=public.read_bytes()
            public_rows=self.api.lines(public_data);public_rows[0]['request']='Tampered input'
            public.write_bytes(self.api.jsonl_bytes(public_rows))
            with self.assertRaises(ValueError):self.api.validate_released_pack(pack,scores)
            public.write_bytes(public_data)
            (pack/'private/mapping.json').unlink()
            reproduced=self.api.create_scores(pack,[first,second],root/'reproduced',review_key=scores/'review-key.json')
            self.assertEqual(reproduced['combined'],json.loads(original_report)['combined'])
            (scores/'reviews/reviewer-01/part-01.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):self.api.validate_released_pack(pack,scores)
