"""CPU-only frozen-input and resumable generation contract tests."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.tiny_aya import evaluate, train, runtime


class FixtureModel:
    training=False
    def generate(self, *, input_ids, attention_mask, generation_config, logits_processor):
        assert not torch.is_grad_enabled()
        for processor in logits_processor:processor(input_ids,torch.zeros((len(input_ids),8)))
        # EOS-only rows are retained as empty, complete responses.
        return torch.cat([input_ids,torch.full((len(input_ids),1),6,dtype=torch.long)],dim=1)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from experiments.tiny_aya.guard import _snapshot
        snapshot,cls.tokenizer_files=_snapshot(train.MODEL,train.REVISION,False)
        cls.tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.snapshot=self.root/'snapshot';self.snapshot.mkdir()
        (self.snapshot/'config.json').write_text(json.dumps(dict(model_type='cohere2',num_hidden_layers=36,hidden_size=2048,vocab_size=262144)))
        self.files={'config.json':train._hash_file(self.snapshot/'config.json'),'weights.safetensors':'fixture-only'}
        self.data={}
        for split,count,languages,prefix in [('train',600,('en','ko','zh'),'t'),('eval_task',100,('en','ko','zh'),'e'),('alignment',24,('en','ko','zh'),'a'),('attacks',100,('en','ko','zh','sw'),'x')]:
            self.data[split]=[dict(id=f'{prefix}{i}:{lang}',family_id=f'{prefix}{i}',lang=lang,instruction='Hello' if i%2 else 'Hello, world!',source_dataset='fixture',source_revision='revision',harm_categories=['fixture'],**({'response':'NEVER MODEL INPUT REFERENCE','labels':[7],'input_ids':[7]} if split in ('train','eval_task') else {})) for i in range(count) for lang in languages]
        self.data['bindings']=dict(data_sha256='fixture-data',tokenizer_files=self.tokenizer_files)
        self.prepared=self.root/'prepared';self.screened=self.root/'screened'
        self.prepared.mkdir();self.screened.mkdir()
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(evaluate.train,'load_training_bundle',side_effect=lambda *args:copy.deepcopy(self.data)))
        stack.enter_context(patch.object(evaluate,'_tokenizer',return_value=(self.tokenizer,self.tokenizer_files)))
        stack.enter_context(patch.object(evaluate,'_checkpoint',side_effect=lambda :(self.snapshot,copy.deepcopy(self.files))))
        stack.enter_context(patch.object(evaluate,'_device',return_value='cpu'))
        self.real_new_model=evaluate._new_model
        self.loader=stack.enter_context(patch.object(evaluate,'_new_model',side_effect=lambda *args:FixtureModel()))
        self.bundle=self.root/'bundle';self.output=self.root/'generation'

    def freeze(self):return evaluate.freeze_bundle(self.prepared,self.screened,self.bundle)
    def generate(self,**kwargs):return evaluate.generate(self.bundle,self.output,arm='base',prepared=self.prepared,screened=self.screened,**kwargs)
    def load(self,**kwargs):return evaluate.load_generation(self.output,self.bundle,self.prepared,self.screened,**kwargs)

    def test_freeze_exact_disjoint_700_preserves_metadata_and_excludes_reference_tokens(self):
        self.freeze();frozen=evaluate.load_bundle(self.bundle,self.prepared,self.screened)
        rows=frozen['rows'];self.assertEqual(len(rows),700)
        self.assertEqual(len({r['id'] for r in rows}),700)
        self.assertEqual(sum(r['split']=='attack' for r in rows),400)
        self.assertEqual(sum(r['split']=='task' for r in rows),300)
        task=rows[-1];self.assertEqual(task['reference'],'NEVER MODEL INPUT REFERENCE')
        self.assertEqual(task['harm_categories'],['fixture'])
        expected=self.tokenizer.apply_chat_template([dict(role='user',content=task['instruction'])],tokenize=True,add_generation_prompt=True,return_dict=False,truncation=False)
        self.assertEqual(task['input_ids'],expected)
        self.assertNotIn('labels',task);self.assertNotIn('response',task)
        self.assertNotIn(str(self.root),json.dumps({k:v for k,v in frozen.items() if k!='tokenizer'}))
        self.assertEqual(frozen['config']['generation']['eos_token_id'],[3,6])
        self.assertEqual(frozen['config']['generation']['pad_token_id'],0)
        self.assertEqual(frozen['config']['generation']['bos_token_id'],2)
        self.assertIn('attention-helpers.py',frozen['config']['sources'])
        self.assertEqual(frozen['config']['implementation_parents'],{'guard_review_commit':'2ce5ffc','training_review_commit':'5ba11a5'})
        with self.assertRaises(FileExistsError):self.freeze()

    def test_partition_overlap_and_input_overflow_fail_before_freeze(self):
        for split,other in [('eval_task','train'),('alignment','attacks')]:
            original=copy.deepcopy(self.data[split]);self.data[split][0]['family_id']=self.data[other][0]['family_id']
            with self.assertRaises(ValueError):self.freeze()
            self.data[split]=original
        self.data['attacks'][0]['instruction']='hello '*5000
        with self.assertRaises(ValueError):self.freeze()

    def test_resume_exact_frozen_batches_and_independent_cpu_consumer(self):
        self.freeze()
        first=self.generate(batch_size=4,max_batches=1)
        self.assertEqual(first['status'],'incomplete');self.assertEqual(first['records'],4)
        loaded=self.load();self.assertEqual(len(loaded['rows']),4)
        prefix=(self.output/'batches.jsonl').read_bytes()
        self.generate(batch_size=4)
        self.assertTrue((self.output/'batches.jsonl').read_bytes().startswith(prefix))
        loaded=self.load();self.assertEqual(loaded['report']['status'],'complete')
        self.assertEqual(len(loaded['rows']),700)
        self.assertEqual([r['item_id'] for r in loaded['rows']],[r['id'] for r in evaluate.load_bundle(self.bundle,self.prepared,self.screened)['rows']])
        self.assertTrue(all(r['empty_response'] for r in loaded['rows']))
        self.assertEqual(self.loader.call_count,2)
        # Archived implementation bytes survive later mutable-checkout edits.
        with patch.object(evaluate,'_source_paths',side_effect=AssertionError('must use archived sources')):self.load()
        with self.assertRaises(ValueError):self.generate(batch_size=1)
        self.assertEqual(self.loader.call_count,2)

    def test_rehashed_float_token_ids_do_not_pass_integer_equality(self):
        self.freeze()
        rows=train._rows(self.bundle/'items.jsonl');rows[0]['input_ids']=[float(i) for i in rows[0]['input_ids']]
        (self.bundle/'items.jsonl').write_text(''.join(train._json(r)+'\n' for r in rows))
        report=train._read(self.bundle/'report.json');report['items_sha256']=train._hash_file(self.bundle/'items.jsonl')
        train._write(self.bundle/'report.json',report)
        manifest=train._read(self.bundle/'manifest.json')
        for name in manifest['files']:manifest['files'][name]['sha256']=train._hash_file(self.bundle/name)
        train._write(self.bundle/'manifest.json',manifest)
        with self.assertRaises(ValueError):evaluate.load_bundle(self.bundle,self.prepared,self.screened)

    def test_duplicate_json_keys_are_rejected_even_when_last_value_is_correct(self):
        self.freeze();self.generate(max_batches=1)
        journal=self.output/'batches.jsonl';blob=journal.read_bytes()
        journal.write_bytes(blob.replace(b'"batch_index":0',b'"batch_index":999,"batch_index":0',1))
        self.rehash_output()
        with self.assertRaises(ValueError):self.load()
        with self.assertRaises(ValueError):self.generate()

    def test_max_batches_must_be_a_strict_incomplete_prefix(self):
        self.freeze()
        for invalid in (True,0,-1,700,701):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):self.generate(max_batches=invalid)
        self.assertFalse(self.output.exists())

    def test_report_status_requires_consistent_failure_type(self):
        self.freeze();self.generate(max_batches=1)
        original=train._read(self.output/'report.json')
        for updates in ({'status':'failed'},{'error_type':'RuntimeError'},{'status':'failed','error_type':7}):
            report={**original,**updates};train._write(self.output/'report.json',report);self.rehash_output()
            with self.subTest(updates=updates),self.assertRaises(ValueError):self.load()
        train._write(self.output/'report.json',original);self.rehash_output();self.load()

    def test_changed_current_data_checkpoint_and_rehashed_config_rejected(self):
        self.freeze();self.generate(max_batches=1)
        self.data['attacks'][0]['instruction']='changed'
        with self.assertRaises(ValueError):self.load()
        self.data['attacks'][0]['instruction']='Hello, world!'
        self.files['config.json']='changed'
        with self.assertRaises(ValueError):self.load()
        self.files['config.json']=train._hash_file(self.snapshot/'config.json')
        config=train._read(self.output/'run-config.json');config['generation']['repetition_penalty']=1.2
        train._write(self.output/'run-config.json',config)
        self.rehash_output()
        with self.assertRaises(ValueError):self.load()

    def rehash_output(self):
        manifest=train._read(self.output/'manifest.json')
        manifest['status']=train._read(self.output/'report.json')['status']
        for name in manifest['files']:manifest['files'][name]['sha256']=train._hash_file(self.output/name)
        train._write(self.output/'manifest.json',manifest)

    def test_corrupt_torn_reordered_and_rehashed_raw_evidence_rejected(self):
        self.freeze();self.generate(batch_size=4,max_batches=2)
        path=self.output/'batches.jsonl';original=path.read_bytes()
        for bad in (original[:-1],original+original.splitlines(keepends=True)[0],b'\n'.join(original.splitlines()[::-1])+b'\n'):
            path.write_bytes(bad)
            with self.assertRaises(ValueError):self.generate(batch_size=4)
        path.write_bytes(original)
        batches=train._rows(path);batches[0]['rows'][0]['generated_ids']=[3]
        path.write_text(''.join(train._json(b)+'\n' for b in batches));self.rehash_output()
        with self.assertRaises(ValueError):self.load()
        path.write_bytes(original);self.rehash_output()
        (self.output/'.writer.lock').write_text('stale')
        with self.assertRaises((RuntimeError,ValueError)):self.generate(batch_size=4)
        self.assertTrue((self.output/'.writer.lock').exists())

    def test_adapter_arm_full_run_data_and_initialization_binding(self):
        self.freeze()
        for arm in ('base','plain','safegrad','unknown'):
            if arm=='base':
                with self.assertRaises(ValueError):evaluate.generate(self.bundle,self.output,arm=arm,prepared=self.prepared,screened=self.screened,training_run=self.root/'missing')
            elif arm=='unknown':
                with self.assertRaises(ValueError):evaluate.generate(self.bundle,self.output,arm=arm,prepared=self.prepared,screened=self.screened)
            else:
                with self.assertRaises(ValueError):evaluate.generate(self.bundle,self.output,arm=arm,prepared=self.prepared,screened=self.screened)
        training=self.root/'training';training.mkdir()
        run=dict(config=dict(arm='plain',settings=copy.deepcopy(train.SETTINGS),max_updates=None,data=copy.deepcopy(self.data['bindings']),checkpoint_files=copy.deepcopy(self.files),initialization={'tensor_sha256':'shared'},initialization_manifest_sha256='init'),report=dict(status='complete',run_kind='full',arm='plain',adapter={'files':{'adapter_model.safetensors':'adapter'}}),adapter=training/'adapter')
        for name in ('run-config.json','report.json'):(training/name).write_text('{}')
        with patch.object(train,'load_training_run',side_effect=lambda *args:copy.deepcopy(run)):
            for key,value in [('arm','safegrad'),('max_updates',1),('data',{}),('settings',{})]:
                old=run['config'][key];run['config'][key]=value
                with self.assertRaises(ValueError):evaluate.generate(self.bundle,self.output,arm='plain',prepared=self.prepared,screened=self.screened,training_run=training,max_batches=1)
                run['config'][key]=old
            run['report']['status']='feasibility'
            with self.assertRaises(ValueError):evaluate.generate(self.bundle,self.output,arm='plain',prepared=self.prepared,screened=self.screened,training_run=training,max_batches=1)
        self.assertFalse(self.loader.called)

    def test_valid_adapter_identity_is_frozen_and_required_by_consumer(self):
        self.freeze();training=self.root/'training';training.mkdir()
        for name in ('run-config.json','report.json'):(training/name).write_text('{}')
        run=dict(config=dict(arm='plain',settings=copy.deepcopy(train.SETTINGS),max_updates=None,data=copy.deepcopy(self.data['bindings']),checkpoint_files=copy.deepcopy(self.files),initialization={'tensor_sha256':'shared'},initialization_manifest_sha256='init'),report=dict(status='complete',run_kind='full',arm='plain',adapter={'files':{'adapter_model.safetensors':'adapter'}}),adapter=training/'adapter')
        with patch.object(train,'load_training_run',side_effect=lambda *args:copy.deepcopy(run)):
            evaluate.generate(self.bundle,self.output,arm='plain',prepared=self.prepared,screened=self.screened,training_run=training,max_batches=1)
            loaded=self.load(training_run=training)
            self.assertEqual(loaded['config']['training']['initialization_tensor_sha256'],'shared')
            self.assertEqual(self.loader.call_args.args[-1],training/'adapter')
            with self.assertRaises(ValueError):self.load()
            run['config']['initialization']['tensor_sha256']='different-shared-init'
            with self.assertRaises(ValueError):self.load(training_run=training)

    def test_fresh_loader_local_only_eager_finite_frozen_bfloat16_model(self):
        class Model:
            def __init__(self):
                self.config=SimpleNamespace(model_type='cohere2',num_hidden_layers=36,hidden_size=2048,vocab_size=262144)
                self.weight=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16))
                self.training=True
            def to(self,device):return self
            def set_attn_implementation(self,backend):self.config._attn_implementation=backend
            def named_modules(self):return [(f'model.layers.{i}.self_attn',SimpleNamespace(config=self.config)) for i in range(36)]
            def named_parameters(self):return [('weight',self.weight)]
            def eval(self):self.training=False;return self
            def _prepare_generation_config(self,config):return config,{}
        from transformers import AutoModelForCausalLM
        generation=evaluate.generation_config();model=Model()
        with patch.object(AutoModelForCausalLM,'from_pretrained',return_value=model) as load:
            result=self.real_new_model(self.snapshot,'cpu',generation,None)
            self.assertIs(result,model);self.assertFalse(model.training);self.assertFalse(model.weight.requires_grad)
            self.assertEqual(load.call_args.kwargs,dict(local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager'))
            self.assertEqual(model.generation_config.to_dict(),generation.to_dict())
        for value,dtype in [(float('nan'),torch.bfloat16),(float('inf'),torch.bfloat16),(1.,torch.float32)]:
            model=Model();model.weight=torch.nn.Parameter(torch.full((2,),value,dtype=dtype))
            with patch.object(AutoModelForCausalLM,'from_pretrained',return_value=model),self.assertRaises(ValueError):
                self.real_new_model(self.snapshot,'cpu',generation,None)

    def test_oom_failure_retains_prefix_no_fallback_and_resumes_same_partition(self):
        self.freeze()
        self.loader.side_effect=lambda *args:FixtureModel()
        with patch.object(runtime,'generate_batch',side_effect=RuntimeError('out of memory')):
            with self.assertRaises(RuntimeError):self.generate(batch_size=4)
        loaded=self.load();self.assertEqual(loaded['report']['status'],'failed');self.assertEqual(loaded['rows'],[])
        self.generate(batch_size=4,max_batches=1)
        self.assertEqual(len(self.load()['rows']),4)

if __name__=='__main__':unittest.main()
