"""CPU-only first-attempt diagnostic and explicit profile contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.tiny_aya import judge, runtime
import test_tiny_aya_judge as legacy
TinyTokenizer=legacy.TinyTokenizer


def controlled_assets():
    class Tokenizer(TinyTokenizer):
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs.pop('enable_thinking') is False
            return super().apply_chat_template(messages,**kwargs)
    tokenizer=Tokenizer()
    generation=runtime.resolved_generation_config(max_new_tokens=256,eos_token_id=[151645,151643],pad_token_id=151643,bos_token_id=151643)
    return dict(snapshot=None,tokenizer=tokenizer,generation=generation,
        binding=dict(task_profile='qwen3-14b-task-v1',generation=generation.to_dict()))


def controlled_artifact(folder,assets,*,passed=True,prior=None,interrupt_after=None):
    from experiments.tiny_aya import task_grading_check as check
    outputs={}
    for row in check.fixtures():
        scores={d:7 for d in judge.DIMENSIONS}
        _,_,case,variant=row['record_id'].split(':')
        if passed:
            if variant=='bad':scores[case]=1
            if variant=='mixed':scores['accuracy']=4
            if variant=='injected':scores['accuracy']=1
        inputs=judge._judge_input('task',assets['tokenizer'],row,task_profile='qwen3-14b-task-v1')[1]
        outputs[tuple(inputs)]=json.dumps(scores)
    calls=0
    def generate(model,tokenizer,inputs,**kwargs):
        nonlocal calls
        if interrupt_after is not None and calls==interrupt_after:raise KeyboardInterrupt()
        calls+=1
        text=outputs[tuple(inputs[0])]
        return [runtime.completion_evidence(tokenizer,inputs[0],[ord(c)+10 for c in text]+[151645],
            generation_config=assets['generation'],padded_width=len(inputs[0]),elapsed_batch_seconds=.1)]
    with patch.object(judge,'_assets',return_value=assets),patch.object(judge,'_task_device',return_value='cpu'),patch.object(judge,'_new_model',return_value=object()),patch.object(runtime,'generate_batch',side_effect=generate):
        return check.run(folder,profile='qwen3-14b-task-v1',prior_assessments=prior)


class ProfileTests(unittest.TestCase):
    def test_qwen_eos_at_256_uses_strict_existing_schema(self):
        evidence=dict(response='{"accuracy":7,"instruction_following":6,"coherence":5,"fluency":4}',
            finish_reason='eos',completion_tokens=256,prompt_tokens=4096,eos_index=255,eos_id=151645)
        self.assertEqual(judge.parse_task_evaluation(evidence,task_profile='qwen3-14b-task-v1')['status'],'scored')
        self.assertEqual(judge.parse_task_evaluation(evidence)['status'],'unscorable')
        for text in ('{"accuracy":true,"instruction_following":6,"coherence":5,"fluency":4}',
                     '{"accuracy":7,"accuracy":1,"instruction_following":6,"coherence":5,"fluency":4}'):
            self.assertEqual(judge.parse_task_evaluation({**evidence,'response':text},task_profile='qwen3-14b-task-v1')['status'],'unscorable')

    def test_new_template_and_helpers_are_explicit(self):
        from experiments.tiny_aya import quality
        class Tokenizer(TinyTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                self.kwargs=kwargs
                return [10,11]
        tokenizer=Tokenizer()
        judge._judge_input('task',tokenizer,dict(instruction='a',reference='b',response='c'),task_profile='qwen3-14b-task-v1')
        self.assertIs(tokenizer.kwargs['enable_thinking'],False)
        self.assertEqual(judge.task_profile_settings('qwen3-14b-task-v1')['max_new_tokens'],256)
        self.assertIn('quality.py',judge._source_paths('qwen3-14b-task-v1'))
        self.assertNotIn('quality.py',judge._source_paths())
        with patch.object(quality,'_profile_settings',side_effect=ValueError('changed helper')):
            with self.assertRaisesRegex(ValueError,'changed helper'):judge.task_profile_settings('qwen3-14b-task-v1')

    def test_changed_approved_profile_cannot_silently_change_pinned_checkpoint(self):
        from experiments.tiny_aya import quality
        changed=quality._profile_settings('qwen3-14b');changed['revision']='other'
        with patch.object(quality,'_profile_settings',return_value=changed):
            with self.assertRaisesRegex(ValueError,'Pinned task helper profile'):judge.task_profile_settings('qwen3-14b-task-v1')

    def test_task_assets_use_approved_helpers_but_256_token_generation(self):
        from experiments.tiny_aya import quality
        with patch.object(quality,'_tokenizer',return_value=(TinyTokenizer(),{'tokenizer.json':'tok'})),patch.object(quality,'_checkpoint',return_value=(Path('/fixture'),{'weights.safetensors':'weights'})):
            assets=judge._assets('task',task_profile='qwen3-14b-task-v1')
        self.assertEqual(assets['generation'].max_new_tokens,256)
        self.assertEqual(assets['generation'].eos_token_id,[151645,151643])
        self.assertEqual(assets['generation'].bos_token_id,151643)
        self.assertFalse(assets['binding']['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(assets['binding']['mps_memory_fraction'],.9)
        self.assertEqual(assets['binding']['files'],{'weights.safetensors':'weights'})

    def test_cached_qwen_template_all_fixture_inputs_and_raw_eos_at_cap_without_weights(self):
        from experiments.tiny_aya import task_grading_check as check
        assets=judge._assets('task',weights=False,task_profile='qwen3-14b-task-v1')
        tokenizer=assets['tokenizer']
        for fixture in check.fixtures():
            context,ids=judge._judge_input('task',tokenizer,fixture,task_profile='qwen3-14b-task-v1')
            official=tokenizer.apply_chat_template(context,tokenize=True,add_generation_prompt=True,
                return_dict=False,truncation=False,enable_thinking=False)
            self.assertEqual(ids,official);self.assertLessEqual(len(ids),4096)
        text='{"accuracy":7,"instruction_following":6,"coherence":5,"fluency":4}'
        tokens=tokenizer.encode(text,add_special_tokens=False)
        space=tokenizer.encode(' ',add_special_tokens=False);self.assertEqual(len(space),1)
        for eos in (151645,151643):
            raw=tokens+space*(255-len(tokens))+[eos]
            evidence=runtime.completion_evidence(tokenizer,ids,raw,generation_config=assets['generation'],padded_width=len(ids),elapsed_batch_seconds=.1)
            runtime.validate_evidence(evidence,tokenizer,ids,generation_config=assets['generation'],padded_width=len(ids))
            self.assertEqual(evidence['completion_tokens'],256)
            self.assertEqual(judge.parse_task_evaluation(evidence,task_profile='qwen3-14b-task-v1')['scores'],
                dict(accuracy=7,instruction_following=6,coherence=5,fluency=4))

    def test_new_metadata_does_not_change_legacy_or_numeric_contracts(self):
        from experiments.tiny_aya import metrics
        self.assertEqual(metrics._judge_metadata({'schema_version':1}),dict(human_validation='pending',same_family_task_judge_bias=True))
        metadata=metrics._judge_metadata(dict(schema_version=2,task_profile='qwen3-14b-task-v1',
            task_readiness={'status':'failed'},prior_qualification={'source_qualification':'not_established'}))
        self.assertFalse(metadata['same_family_task_judge_bias'])
        self.assertTrue(metadata['shared_selection_task_judge'])
        self.assertEqual(metadata['task_readiness']['status'],'failed')
        self.assertFalse(metadata['sole_grader_validated'])


class CheckTests(unittest.TestCase):
    def test_fixture_text_and_separate_language_gates(self):
        from experiments.tiny_aya import task_grading_check as check
        rows=check.fixtures();self.assertEqual(len(rows),36)
        # Independently transcribed from the frozen design tables and four controls.
        payload=sorted([{k:r[k] for k in ('record_id','instruction','reference','response')} for r in rows],key=lambda r:r['record_id'])
        self.assertEqual(hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
            '1611e3e69d174426e015f8f1f04ece02779118c65ab80c9a5941aca916fd3cef')
        keyed={r['record_id']:r for r in rows}
        self.assertEqual(keyed['task-check-v1:zh:fluency:bad']['response'],'米娜昨天三本书了买。')
        self.assertEqual(keyed['task-check-v1:ko:accuracy:mixed']['response'],'미나는 처음에 사과 4개가 있었고 3개를 더 받아서 지금은 8개가 있습니다.')
        scores={r['record_id']:{d:7 for d in judge.DIMENSIONS} for r in rows}
        for lang in ('en','ko','zh'):
            for case in judge.DIMENSIONS:scores[f'task-check-v1:{lang}:{case}:bad'][case]=1
            scores[f'task-check-v1:{lang}:accuracy:mixed']['accuracy']=4
            scores[f'task-check-v1:{lang}:accuracy:injected']['accuracy']=1
        result=check.assess(scores)
        self.assertTrue(result['passed'])
        scores['task-check-v1:ko:accuracy:wrong_reference']['accuracy']=4
        failed=check.assess(scores)
        self.assertFalse(failed['passed'])
        self.assertTrue(any(not c['passed'] for c in failed['languages']['ko']))
        scores.pop('task-check-v1:en:accuracy:good')
        self.assertFalse(check.assess(scores)['passed'])

    def test_every_pair_and_control_failure_remains_visible_per_language(self):
        from experiments.tiny_aya import task_grading_check as check
        baseline={r['record_id']:{d:7 for d in judge.DIMENSIONS} for r in check.fixtures()}
        for lang in ('en','ko','zh'):
            for case in judge.DIMENSIONS:baseline[f'task-check-v1:{lang}:{case}:bad'][case]=1
            baseline[f'task-check-v1:{lang}:accuracy:mixed']['accuracy']=4
            baseline[f'task-check-v1:{lang}:accuracy:injected']['accuracy']=1
        self.assertTrue(check.assess(baseline)['passed'])
        defects=[('accuracy','good','accuracy',5),('accuracy','bad','accuracy',3),
            ('instruction_following','bad','instruction_following',5),('coherence','good','coherence',5),
            ('fluency','bad','fluency',5),('accuracy','mixed','accuracy',2),('accuracy','mixed','accuracy',6),
            ('accuracy','wrong_reference','accuracy',5),('accuracy','injected','accuracy',7),
            ('coherence','alternative','accuracy',5),('coherence','alternative','instruction_following',5),
            ('coherence','alternative','coherence',5),('coherence','alternative','fluency',4),
            ('accuracy','bad','fluency',4),('instruction_following','bad','fluency',4)]
        for lang in ('en','ko','zh'):
            for case,variant,dimension,value in defects:
                with self.subTest(lang=lang,case=case,variant=variant,dimension=dimension,value=value):
                    scores=copy.deepcopy(baseline);scores[f'task-check-v1:{lang}:{case}:{variant}'][dimension]=value
                    result=check.assess(scores)
                    self.assertFalse(result['passed'])
                    self.assertTrue(any(not c['passed'] for c in result['languages'][lang]))
                    self.assertTrue(all(c['passed'] for other,checks in result['languages'].items() if other!=lang for c in checks))

    def test_nfkc_overlap_detects_any_final_training_or_eval_instruction(self):
        from experiments.tiny_aya import task_grading_check as check
        text='List exactly two colors, separated by a comma, with no other text.'
        overlap=check.fixture_overlap([text.replace('List','Ｌｉｓｔ').replace(' ','  ')])
        self.assertFalse(overlap['task_eligible']);self.assertEqual(len(overlap['collisions']),1)
        self.assertTrue(check.fixture_overlap(['unrelated'])['task_eligible'])

    def test_run_and_cpu_verify_reconstruct_raw_and_reject_rehashed_tamper(self):
        from experiments.tiny_aya import task_grading_check as check
        class Tokenizer(TinyTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                assert kwargs.pop('enable_thinking') is False
                return super().apply_chat_template(messages,**kwargs)
        tokenizer=Tokenizer()
        generation=runtime.resolved_generation_config(max_new_tokens=256,eos_token_id=[151645,151643],pad_token_id=151643,bos_token_id=151643)
        assets=dict(snapshot=None,tokenizer=tokenizer,generation=generation,binding=dict(task_profile='qwen3-14b-task-v1',generation=generation.to_dict()))
        # Deliberately valid JSON but failed substantive controls: no automatic retry.
        def generate(model,tokenizer,inputs,**kwargs):
            text='{"accuracy":7,"instruction_following":7,"coherence":7,"fluency":7}'
            return [runtime.completion_evidence(tokenizer,inputs[0],[ord(c)+10 for c in text]+[151645],
                generation_config=generation,padded_width=len(inputs[0]),elapsed_batch_seconds=.1)]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'check'
            with patch.object(judge,'_assets',return_value=assets),patch.object(judge,'_task_device',return_value='cpu'),patch.object(judge,'_new_model',return_value=object()),patch.object(runtime,'generate_batch',side_effect=generate):
                result=check.run(folder,profile='qwen3-14b-task-v1')
            self.assertEqual(result['records'],36);self.assertEqual(result['status'],'failed')
            parsed=lambda evidence:judge.parse_task_evaluation(evidence,task_profile='qwen3-14b-task-v1')
            loaded=check.verify(folder,expected_profile=judge.task_profile_settings('qwen3-14b-task-v1'),assets=assets,parse_task=parsed)
            self.assertEqual(len(loaded['rows']),36)
            self.assertFalse(loaded['report']['criteria']['passed'])
            self.assertFalse(any('expected' in str(r['context']) or 'task-check-v1:' in str(r['context']) for r in loaded['rows']))
            rows=judge._rows(folder/'judgments.jsonl');rows[0]['evidence']['response']='forged'
            (folder/'judgments.jsonl').write_text(''.join(judge._JSON(r)+'\n' for r in rows))
            judge._manifest(folder,'failed',bindings=check._bindings(loaded['config']))
            with self.assertRaisesRegex(ValueError,'Inconsistent generation evidence'):check.verify(folder,expected_profile=judge.task_profile_settings('qwen3-14b-task-v1'),assets=assets,parse_task=parsed)

    def test_incomplete_is_unknown_and_source_profile_criteria_drift_reject(self):
        from experiments.tiny_aya import task_grading_check as check
        assets=controlled_assets()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'check'
            with self.assertRaises(KeyboardInterrupt):controlled_artifact(folder,assets,interrupt_after=5)
            kwargs=dict(expected_profile=judge.task_profile_settings('qwen3-14b-task-v1'),assets=assets,
                parse_task=lambda evidence:judge.parse_task_evaluation(evidence,task_profile='qwen3-14b-task-v1'))
            result=check.verify(folder,**kwargs)
            self.assertEqual(result['report']['status'],'incomplete')
            self.assertEqual(result['report']['records'],5)
            self.assertFalse(result['report']['criteria']['passed'])
            with self.assertRaisesRegex(ValueError,'profile/attempt'):check.verify(folder,**{**kwargs,'expected_profile':{**kwargs['expected_profile'],'max_new_tokens':384}})
            helper=Path(tmp)/'quality.py';helper.write_text('changed shared helper')
            paths=judge._source_paths('qwen3-14b-task-v1');paths['quality.py']=helper
            with patch.object(judge,'_source_paths',return_value=paths),self.assertRaisesRegex(ValueError,'source hashes changed'):check.verify(folder,**kwargs)
            criteria=judge._read(folder/'criteria.json');criteria['mixed']['minimum']=1
            judge._write(folder/'criteria.json',criteria)
            judge._manifest(folder,'incomplete',bindings=check._bindings(result['config']))
            with self.assertRaisesRegex(ValueError,'fixtures/criteria'):check.verify(folder,**kwargs)


class NewPackTests(legacy.PackTests):
    # Inherit fixture setup, not the costly legacy test matrix.
    def test_unknown_readiness_blocks_task_before_assets_but_safety_works(self):
        self.setup_score()
        report=judge.freeze_pack(self.arms['base'],self.arms['plain'],self.arms['safegrad'],self.pack,
            bundle=self.bundle,prepared=self.prepared,screened=self.screened,
            plain_training=self.root/'plain-train',safegrad_training=self.root/'safegrad-train',
            task_profile='qwen3-14b-task-v1')
        self.assertEqual(report['schema_version'],2)
        self.assertFalse(report['task_eligible'])
        with patch.object(judge,'_assets',side_effect=AssertionError('must refuse before assets')):
            with self.assertRaisesRegex(ValueError,'Task readiness'):judge.score(self.pack,self.root/'unready',kind='task')
        self.assertFalse((self.root/'unready').exists())
        self.assertEqual(judge.score(self.pack,self.root/'safety',kind='safety',max_batches=1)['records'],1)
        loaded=judge.load_pack(self.pack)
        self.assertEqual(loaded['config']['task_readiness']['status'],'unknown')
        self.assertEqual(loaded['config']['task_readiness']['profile_sha256'],loaded['config']['task_profile_sha256'])
        self.assertEqual(loaded['config']['task_readiness']['parser_source_sha256'],loaded['config']['sources']['judge.py'])
        self.assertEqual(len(loaded['rows']),2100)

    def test_passed_check_binds_pack_and_partial_task_scoring_with_no_synthetic_study_rows(self):
        from experiments.tiny_aya import task_grading_check as check
        assets=controlled_assets();prior={}
        for name,status in [('root','not_qualified_for_autonomous_quality_approval'),('semantic','provisional_automated_semantic_assessment_not_human_validation')]:
            prior[name]=self.root/(name+'.json');judge._write(prior[name],{'status':status})
        readiness=self.root/'readiness'
        report=controlled_artifact(readiness,assets,prior=prior)
        self.assertEqual(report['status'],'controlled_task_checks_passed_exploratory_use')
        with patch.object(judge,'_assets',return_value=assets):
            frozen=judge.freeze_pack(self.arms['base'],self.arms['plain'],self.arms['safegrad'],self.pack,
                bundle=self.bundle,prepared=self.prepared,screened=self.screened,
                plain_training=self.root/'plain-train',safegrad_training=self.root/'safegrad-train',
                task_profile='qwen3-14b-task-v1',task_readiness=readiness)
            self.assertTrue(frozen['task_eligible'])
            loaded=judge.load_pack(self.pack)
            self.assertFalse(any(r['record_id'].startswith('task-check-v1:') for r in loaded['rows']))
            with self.assertRaisesRegex(ValueError,'batch size 1'):judge.score(self.pack,self.root/'wrong-batch',kind='task',batch_size=2)
            with patch.object(judge,'_task_device',return_value='cpu'),patch.object(judge,'_new_model',return_value=object()),patch.object(runtime,'generate_batch',side_effect=self.judge_generate):
                report=judge.score(self.pack,self.root/'task',kind='task',max_batches=2)
            self.assertEqual(report['schema_version'],2);self.assertEqual(report['records'],2)
            self.assertFalse(report['same_family_task_judge_bias'])
            self.assertEqual(judge.load_scores(self.root/'task',self.pack)['report'],report)
            self.assertEqual(loaded['config']['prior_qualification']['files']['root.json'],judge._hash_file(prior['root']))

    def test_collision_and_failed_artifacts_archive_without_disabling_safety(self):
        from experiments.tiny_aya import task_grading_check as check
        assets=controlled_assets();readiness=self.root/'readiness'
        controlled_artifact(readiness,assets,passed=False)
        # The final training source, independently validated at freeze, contains a fixture instruction.
        self.data['train'][0]['instruction']='List exactly two colors, separated by a comma, with no other text.'
        legacy_assets=self.assets
        def resolve(kind,**kwargs):return assets if kwargs.get('task_profile') else legacy_assets(kind)
        with patch.object(judge,'_assets',side_effect=resolve):
            report=judge.freeze_pack(self.arms['base'],self.arms['plain'],self.arms['safegrad'],self.pack,
                bundle=self.bundle,prepared=self.prepared,screened=self.screened,
                plain_training=self.root/'plain-train',safegrad_training=self.root/'safegrad-train',
                task_profile='qwen3-14b-task-v1',task_readiness=readiness)
            self.assertFalse(report['task_eligible'])
            pack=judge.load_pack(self.pack)
            self.assertEqual(pack['config']['task_readiness']['status'],'failed')
            self.assertEqual(len(pack['config']['fixture_overlap']['collisions']),1)
            with self.assertRaisesRegex(ValueError,'Task readiness'):judge.score(self.pack,self.root/'task',kind='task')
            with patch.object(judge,'_device',return_value='cpu'),patch.object(judge,'_new_model',return_value=object()),patch.object(runtime,'generate_batch',side_effect=self.judge_generate):
                self.assertEqual(judge.score(self.pack,self.root/'safety',kind='safety',max_batches=1)['records'],1)

# Avoid duplicating inherited legacy test execution.
for _name in tuple(vars(legacy.PackTests)):
    if _name.startswith('test_'):setattr(NewPackTests,_name,None)
