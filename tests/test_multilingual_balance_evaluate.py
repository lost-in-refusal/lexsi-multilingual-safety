"""Synthetic checks for the balanced-study generation boundary."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class BalanceEvaluationTests(unittest.TestCase):
    def api(self):
        from experiments.language_balance import evaluate
        return evaluate

    def test_immutable_generation_helpers_and_no_reference_suffix(self):
        from experiments.multilingual_repair import evaluate as old
        ev = self.api()
        self.assertIs(ev.generate_one, old.generate_one)
        self.assertIs(ev.prompt_inputs, old.prompt_inputs)
        self.assertIs(ev.termination, old.termination)
        task = dict(category='task', reference='SECRET', input_ids=[11,12,90,2],
                    attention_mask=[1]*4, prompt_tokens=2, sequence_tokens=4,
                    labels=[-100,-100,90,2])
        self.assertEqual(ev.prompt_inputs(task), ([11,12], [1,1]))
        bad = copy.deepcopy(task); bad['prompt_tokens'] = 4
        with self.assertRaises(ValueError): ev.prompt_inputs(bad)
        self.assertFalse(ev.termination([7,2],[2],2)['cap_reached'])

    def test_frozen_development_coverage_and_cap(self):
        ev=self.api()
        rows=[dict(id=f'{l}-{c}-{i}', language=l, category=c, family_id=f'f{i}')
              for l in ('en','ko','zh') for c in ('task','disallowed') for i in range(12)]
        bundle={'rows':rows,'config':{'generation':{'max_new_tokens':512}}}
        ev.validate_development(bundle)
        for changed in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError): ev.validate_development(dict(bundle,rows=changed))
        with self.assertRaises(ValueError):
            ev.validate_development(dict(bundle,config={'generation':{'max_new_tokens':256}}))

    def test_inference_checkpoint_bytes_match_training(self):
        import tempfile
        from pathlib import Path
        ev=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            snapshot=Path(tmp);file=snapshot/'model.safetensors';file.write_bytes(b'synthetic-weights')
            run={'identity':{'base_checkpoint_files':{'model.safetensors':ev.sha256(file.read_bytes())}}}
            ev.validate_checkpoint(snapshot,run)
            file.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checkpoint'):ev.validate_checkpoint(snapshot,run)

    def test_actual_training_artifacts_revalidated_before_evaluation(self):
        from tests import test_multilingual_balance_train as fixture
        case=fixture.BalanceTrainingTests('runTest');case.setUp();self.addCleanup(case.doCleanups)
        run=case.make_run();ev=self.api()
        bundle={k:case.study['bundle'][k] for k in ('config','manifest_sha256')}
        loaded=ev.load_training_run(run,bundle,case.study)
        self.assertEqual(loaded['identity']['arm'],'control')
        for file in (run/'steps.jsonl',run/'adapter/adapter_model.safetensors',case.root/'init/adapter_model.safetensors'):
            data=file.read_bytes();file.write_bytes(data+b' ')
            with self.subTest(file=file),self.assertRaises(ValueError):ev.load_training_run(run,bundle,case.study)
            file.write_bytes(data)
        case.report['training_order'][0]='ko';case.save_report()
        with self.assertRaises(ValueError):ev.load_training_run(run,bundle,case.study)

    def test_training_loader_rechecks_truthful_study_identity(self):
        ev=self.api()
        from pathlib import Path
        import tempfile, json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=b'{"arm":"control"}\n'; (root/'report.json').write_bytes(raw)
            report={'arm':'control','run_kind':'fixed_schedule','status':'passed_training',
                    'training_method':'safegrad','optimizer_steps':1000,'expected_steps':1000,
                    'study_manifest_sha256':'a'*64,'initialization':{'manifest_sha256':'b'*64}}
            with patch('experiments.language_balance.train.load_training_run', return_value=(root/'adapter',report)):
                loaded=ev.load_training_run(root,{}, {'manifest_sha256':'a'*64})
                self.assertEqual(loaded['identity']['arm'],'control')
                self.assertFalse(loaded['identity']['feasibility_only'])
                for key,value in [('optimizer_steps',999),('run_kind','feasibility'),('arm','plain'),
                                  ('training_method','plain'),('study_manifest_sha256','c'*64)]:
                    with self.subTest(key=key):
                        bad=dict(report,**{key:value})
                        with patch('experiments.language_balance.train.load_training_run',return_value=(root/'adapter',bad)):
                            with self.assertRaises(ValueError): ev.load_training_run(root,{}, {'manifest_sha256':'a'*64})
