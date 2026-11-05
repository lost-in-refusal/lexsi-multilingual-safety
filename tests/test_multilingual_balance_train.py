"""CPU-only initialization identity and truthful training-run validation."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM
from experiments.multilingual_repair import train as old


class BalanceTrainingTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('experiments.language_balance.train')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source={'original_config_sha256':'a'*64,'study_manifest_sha256':'b'*64,
                     'model':old.MODEL,'model_revision':old.REVISION,'base_checkpoint_files':{'model.safetensors':'c'*64}}

    def model(self):
        torch.manual_seed(17)
        model=get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=23,hidden_size=16,intermediate_size=24,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)),
            LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules=sorted(old.TARGETS),task_type='CAUSAL_LM'))
        model.peft_config['default'].base_model_name_or_path=old.MODEL
        model.peft_config['default'].revision=old.REVISION
        return model

    def test_capture_load_restores_exact_shared_tensor_bytes(self):
        model=self.model();directory=self.root/'initialization'
        record=self.api.capture_initialization(model,directory,self.source)
        original={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:p.add_(1.)
        loaded=self.api.load_initialization(model,directory,self.source)
        self.assertEqual(record,loaded)
        self.assertTrue(all(torch.equal(original[n],p) for n,p in model.named_parameters() if p.requires_grad))
        self.assertTrue(all(value['dtype']=='torch.float32' for value in loaded['tensors'].values()))
        self.assertTrue(loaded['zero_lora_B'])
        with self.assertRaises(ValueError):self.api.capture_initialization(model,directory,self.source)

    def test_capture_rejects_trained_and_nonfinite_initialization(self):
        for value in (0.001,float('nan')):
            model=self.model()
            with torch.no_grad():next(p for n,p in model.named_parameters() if '.lora_B.' in n).fill_(value)
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.api.capture_initialization(model,self.root/'bad',self.source)
        self.assertFalse((self.root/'bad').exists())

    def test_missing_tampered_and_wrong_source_initialization_fail(self):
        model=self.model();directory=self.root/'initialization'
        with self.assertRaises(ValueError):self.api.load_initialization(model,directory,self.source)
        self.api.capture_initialization(model,directory,self.source)
        wrong=copy.deepcopy(self.source);wrong['original_config_sha256']='d'*64
        with self.assertRaises(ValueError):self.api.load_initialization(model,directory,wrong)
        weights=directory/'adapter_model.safetensors';weights.write_bytes(weights.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'hash'):self.api.load_initialization(model,directory,self.source)

    def test_initialization_config_tamper_rejected_even_after_file_rehash(self):
        model=self.model();directory=self.root/'initialization'
        self.api.capture_initialization(model,directory,self.source)
        config=json.loads((directory/'adapter_config.json').read_bytes());config['lora_alpha']=99
        old.write_json(directory/'adapter_config.json',config)
        manifest=json.loads((directory/'manifest.json').read_bytes())
        manifest['files']['adapter_config.json']=old.sha_bytes((directory/'adapter_config.json').read_bytes());old.write_json(directory/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'config'):self.api.load_initialization(model,directory,self.source)

    def test_tensor_manifest_rejects_self_consistent_nonzero_B(self):
        from safetensors.torch import load_file,save_file
        model=self.model();directory=self.root/'initialization'
        self.api.capture_initialization(model,directory,self.source)
        state=load_file(directory/'adapter_model.safetensors')
        state[next(k for k in state if '.lora_B.' in k)].fill_(1.)
        save_file(state,directory/'adapter_model.safetensors')
        manifest=json.loads((directory/'manifest.json').read_bytes())
        manifest['files']['adapter_model.safetensors']=old.sha_bytes((directory/'adapter_model.safetensors').read_bytes())
        old.write_json(directory/'manifest.json',manifest)
        with self.assertRaises(ValueError):self.api.load_initialization(model,directory,self.source)

    def make_run(self):
        model=self.model();model.peft_config['default'].base_model_name_or_path=old.MODEL;model.peft_config['default'].revision=old.REVISION
        config=json.loads((old.ROOT/'experiments/multilingual_repair/config.json').read_bytes())
        configbytes=(json.dumps(config)+'\n').encode()
        pool=[dict(id=language,language=language,split='train',supervised_tokens=2) for language in ('en','ko','zh')]
        order=['en']*450+['ko']*100+['zh']*450
        self.study={'pool_by_id':{r['id']:r for r in pool},'schedules':{'control':order,'balanced':['en']*333+['ko']*334+['zh']*333},
            'alignment_order':['align']*1000,'manifest_sha256':'b'*64,'manifest':{'files':{'training-config.json':'placeholder'}},
            'spec':{'original_config_sha256':old.sha_bytes(configbytes)}}
        self.study['bundle']={'config':config,'manifest_sha256':'e'*64,'manifest_bytes':b'{"files":{}}\n',
            'manifest':{'files':{'config.json':old.sha_bytes(configbytes)}},'blobs':{'config.json':configbytes},
            'calibration':[dict(id=f'cal-{language}-{i}',language=language,supervised_tokens=2) for language in ('en','ko','zh') for i in range(6)]}
        self.study['bundle']['manifest_sha256']=old.sha_bytes(self.study['bundle']['manifest_bytes'])
        run=self.root/'run';run.mkdir();self.run=run
        specbytes=(json.dumps(self.study['spec'])+'\n').encode()
        self.study['manifest']['files']['training-config.json']=old.sha_bytes(specbytes)
        manifestbytes=(json.dumps(self.study['manifest'])+'\n').encode();self.study['manifest_sha256']=old.sha_bytes(manifestbytes)
        (run/'study-manifest.json').write_bytes(manifestbytes)
        source=self.api.initialization_config(self.study,self.source['base_checkpoint_files'])
        init=self.api.capture_initialization(model,self.root/'init',source)
        with torch.no_grad():
            for n,p in model.named_parameters():
                if '.lora_B.' in n:p.fill_(0.01)
        model.save_pretrained(run/'adapter',safe_serialization=True,save_embedding_layers=False)
        from peft import get_peft_model_state_dict
        tensors=self.api.tensor_metadata(get_peft_model_state_dict(model, save_embedding_layers=False),require_zero_b=False)
        logs=[]
        for index,identity in enumerate(order,1):
            logs.append(dict(step=index,sample_id=identity,split='train',language=identity,supervised_tokens=2,
                alignment_id='align',task_loss=1.,alignment_loss=0.,gradient_norm_before_clip=1.,finite_adapter_weights=True,
                gradient_statistics=dict(dot=0.,task_norm=1.,alignment_norm=0.,conflict=False,zero_alignment=True,projection_coefficient=0.,rho=1.,cosine=None)))
        (run/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs))
        for name,data in [('training-config.json',specbytes),('original-training-config.json',configbytes),
            ('prepared-manifest.json',self.study['bundle']['manifest_bytes'])]: (run/name).write_bytes(data)
        for name,path in [('training-source.py',Path(self.api.__file__)),('study-source.py',Path(importlib.import_module('experiments.language_balance.study').__file__)),('numerical-source.py',Path(old.__file__))]:
            (run/name).write_bytes(path.read_bytes())
        cal=old.aggregate_calibration([dict(r,loss=1.) for r in self.study['bundle']['calibration']])
        self.report=dict(status='passed_training',stage='complete',run_kind='fixed_schedule',training_method='safegrad',arm='control',
            model=old.MODEL,model_revision=old.REVISION,attention_implementation='eager',
            study_manifest_sha256=self.study['manifest_sha256'],prepared_manifest_sha256=self.study['bundle']['manifest_sha256'],
            original_config_sha256=old.sha_bytes(configbytes),config_sha256=old.sha_bytes(specbytes),
            training_config=self.study['spec'],training_order=order,alignment_order=['align']*1000,expected_steps=1000,optimizer_steps=1000,
            initialization=init,base_checkpoint_files=self.source['base_checkpoint_files'],source_hashes=self.api.source_hashes(),environment=self.api.verify_environment(),
            fresh_optimizer_state=True,saved_adapter_reload_exact=True,all_base_parameters_frozen=True,frozen_parameter_samples_unchanged=True,
            changed_adapter_tensors=7,maximum_adapter_absolute_change=0.009999999776482582,finite_adapter_weights=True,
            memory_preflight=dict(status='passed',optimizer_steps=0,adapter_state_unchanged=True,initial_reference_equivalence_verified=True),
            initial_reference_preflight=dict(status='passed',initial_lora_B_zero=True,reference_logits_equal=True,student_requires_grad=True,
                initial_alignment_loss=0.,initial_alignment_gradients_zero=True),
            attention_backend=dict(requested='eager',effective='eager',attention_modules=1),
            calibration_before=cal,calibration_after=cal,calibration_ids=[r['id'] for r in self.study['bundle']['calibration']],
            adapter_tensors=tensors,adapter_files={p.name:old.sha_bytes(p.read_bytes()) for p in (run/'adapter').iterdir() if p.is_file()},
            steps_sha256=old.sha_bytes((run/'steps.jsonl').read_bytes()),training_script_sha256=old.sha_bytes(Path(self.api.__file__).read_bytes()),
            optimizer=dict(name='AdamW',learning_rate=1e-5,weight_decay=0.,betas=[.9,.999],epsilon=1e-8,foreach=False,batch_size=1,gradient_accumulation_steps=1,max_gradient_norm=1.))
        self.save_report()
        return run

    def save_report(self):(self.run/'report.json').write_text(json.dumps(self.report))

    def test_full_run_loader_accepts_exact_complete_artifacts(self):
        run=self.make_run();adapter,report=self.api.load_training_run(run,self.study)
        self.assertEqual(adapter,run/'adapter');self.assertEqual(report['arm'],'control')

    def test_full_run_loader_rejects_partial_mislabeled_and_nonfinite(self):
        self.make_run();original=copy.deepcopy(self.report)
        for changes in ({'optimizer_steps':2},{'run_kind':'feasibility'},{'training_method':'plain'},
                        {'status':'failed'},{'environment':{}},{'training_order':['en']*1000},{'maximum_adapter_absolute_change':float('nan')},
                        {'saved_adapter_reload_exact':False},{'source_hashes':{}},{'original_config_sha256':'0'*64}):
            self.report={**copy.deepcopy(original),**changes};self.save_report()
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.api.load_training_run(self.run,self.study)

    def test_full_run_loader_rejects_changed_adapter_logs_and_initialization(self):
        self.make_run()
        for path in (self.run/'adapter/adapter_model.safetensors',self.run/'steps.jsonl',self.root/'init/adapter_model.safetensors'):
            original=path.read_bytes();path.write_bytes(original+b' ')
            with self.subTest(path=path),self.assertRaises(ValueError):self.api.load_training_run(self.run,self.study)
            path.write_bytes(original)

    def test_rehashed_log_missing_numerical_statistics_is_rejected(self):
        self.make_run();path=self.run/'steps.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[1]['gradient_statistics']={'rho':1.}
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.report['steps_sha256']=old.sha_bytes(path.read_bytes());self.save_report()
        with self.assertRaisesRegex(ValueError,'diagnostic|statistic'):self.api.load_training_run(self.run,self.study)

    def test_two_step_prefix_is_feasibility_and_full_requires1000(self):
        study={'schedules':{'control':['i']*1000,'balanced':['j']*1000},'alignment_order':['a']*1000}
        self.assertEqual(self.api.select_schedule(study,'control',2),(['i','i'],['a','a'],'feasibility'))
        self.assertEqual(self.api.select_schedule(study,'balanced',None)[2],'fixed_schedule')
        for invalid in (0,1,1000,True):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):self.api.select_schedule(study,'control',invalid)

    def test_failed_cli_serializes_failure_without_loading_weights(self):
        result=self.api.main(['--prepared',str(self.root/'missing'),'--study',str(self.root/'missing-study'),
                             '--initialization',str(self.root/'init'),'--arm','balanced','--max-steps','2','--output',str(self.root/'run')])
        self.assertEqual(result,1)
        report=json.loads((self.root/'run/report.json').read_bytes())
        self.assertEqual(report['status'],'failed');self.assertEqual(report['training_method'],'safegrad')
        self.assertEqual(report['arm'],'balanced');self.assertEqual(report['run_kind'],'feasibility')
        self.assertFalse(report['model_weights_loaded'])


if __name__=='__main__':unittest.main()
