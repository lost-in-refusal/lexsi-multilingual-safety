"""CPU acceptance checks for Tiny Aya's frozen accumulation and artifact contract."""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.tiny_aya import train


def bundle():
    return dict(train=[dict(id=f't{i}:{lang}', family_id=f't{i}', lang=lang) for i in range(600) for lang in ('en','ko','zh')],
                eval_task=[dict(id=f'e{i}:{lang}', family_id=f'e{i}', lang=lang) for i in range(100) for lang in ('en','ko','zh')],
                alignment=[dict(id=f'a{i}:{lang}', family_id=f'a{i}', lang=lang) for i in range(24) for lang in ('en','ko','zh')],
                attacks=[dict(id=f'x{i}:{lang}', family_id=f'x{i}', lang=lang) for i in range(100) for lang in ('en','ko','zh','sw')])


class PlanTests(unittest.TestCase):
    def test_three_epochs_flush_eight_and_cycle_alignment(self):
        plan = train.freeze_training_plan(bundle())
        self.assertEqual(len(plan['updates']),171)
        self.assertEqual([len(u['task_ids']) for u in plan['updates']], ([32]*56+[8])*3)
        self.assertEqual(Counter(i for u in plan['updates'] for i in u['task_ids']), Counter({r['id']:3 for r in bundle()['train']}))
        self.assertEqual(set(Counter(i for u in plan['updates'] for i in u['alignment_ids']).values()),{75})
        self.assertEqual(plan['updates'][0]['learning_rate'],2.5e-5)
        self.assertAlmostEqual(plan['updates'][-1]['learning_rate'],1.2e-6,places=18)
        self.assertAlmostEqual(plan['updates'][85]['learning_rate'],1.31e-5,places=18)
        self.assertEqual(plan,train.freeze_training_plan(bundle()))
        plan['settings']['accumulation']=8
        self.assertEqual(train.freeze_training_plan(bundle())['settings']['accumulation'],32)
        plan=train.freeze_training_plan(bundle())
        self.assertEqual(len(train.select_updates(plan,1)),1)
        self.assertEqual(train.select_updates(plan,1)[0],plan['updates'][0])
        for invalid in (0,171,True,-1):
            with self.assertRaises(ValueError):train.select_updates(plan,invalid)

    def test_rejects_cross_language_duplicate_and_overlapping_families(self):
        for split in ('train','alignment'):
            bad=bundle();bad[split][0]['lang']='sw'
            with self.assertRaises(ValueError):train.freeze_training_plan(bad)
        bad=bundle();bad['alignment'][0]['family_id']='x0'
        with self.assertRaises(ValueError):train.freeze_training_plan(bad)
        bad=bundle();bad['train'][0]['id']=bad['train'][1]['id']
        with self.assertRaises(ValueError):train.freeze_training_plan(bad)

    def test_equal_example_streaming_means_use_actual_count(self):
        for count in (32,8):
            p=torch.tensor([.5,-.2],requires_grad=True)
            examples=[(p-torch.tensor([float(i),float(i+1)])).square().mean() for i in range(count)]
            expected=torch.autograd.grad(torch.stack(examples).mean(),p,retain_graph=True)[0]
            actual=train.accumulate_gradient_mean((torch.autograd.grad(loss,p,retain_graph=True) for loss in examples),count=count)
            torch.testing.assert_close(actual[0],expected)
            self.assertEqual(actual[0].dtype,torch.float32);self.assertFalse(actual[0].requires_grad)
        for rows,count in (([],0),([[None]],1),([[torch.tensor(float('nan'))]],1),([[torch.tensor(1.)]],2)):
            with self.assertRaises(ValueError):train.accumulate_gradient_mean(iter(rows),count=count)

    def test_projection_happens_after_means(self):
        tasks=[[torch.tensor([-3.,1.])],[torch.tensor([1.,1.])]]
        aligns=[[torch.tensor([1.,0.])],[torch.tensor([-1.,1.])]]
        actual,stats=train.finalize_gradients(train.accumulate_gradient_mean(iter(tasks),count=2),train.accumulate_gradient_mean(iter(aligns),count=2))
        torch.testing.assert_close(actual[0],torch.tensor([-1.,1.5]))
        self.assertEqual(stats['projection_coefficient'],0.)
        projected=[train.finalize_gradients(t,a)[0][0] for t,a in zip(tasks,aligns)]
        self.assertFalse(torch.equal(actual[0],sum(projected)/2))


class TokenTests(unittest.TestCase):
    def test_exact_tiny_suffix_and_alignment_user_only(self):
        from transformers import AutoTokenizer
        from experiments.tiny_aya.guard import _snapshot, tokenize_row
        path,_=_snapshot(train.MODEL,train.REVISION,False)
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        row=tokenize_row(tokenizer,dict(id='t:ko',family_id='t',lang='ko',instruction='안녕하세요',response='안녕하세요!'))
        actual=train.verify_task_row(tokenizer,row)
        self.assertEqual(actual['input_ids'][-2:],[261001,6])
        self.assertEqual(actual['attention_mask'],[1]*actual['sequence_tokens'])
        alignment=train.tokenize_alignment(tokenizer,dict(id='a:zh',family_id='a',lang='zh',instruction='你好'))
        self.assertNotIn('response',alignment)
        prompt=tokenizer.apply_chat_template([dict(role='user',content='你好')],tokenize=True,add_generation_prompt=True,return_dict=False,truncation=False)
        self.assertEqual(alignment['input_ids'],prompt)
        for key,value in [('labels',[-100]*len(row['labels'])),('supervised_tokens',1),('lang','sw'),('labels',[float(i) for i in row['labels']]),('attention_mask',[0]*len(row['input_ids']))]:
            bad={**row,key:value}
            with self.assertRaises(ValueError):train.verify_task_row(tokenizer,bad)
        with self.assertRaises(ValueError):train.tokenize_alignment(tokenizer,{**alignment,'lang':'sw'})
        with self.assertRaises(ValueError):train.tokenize_alignment(tokenizer,{**alignment,'response':'leak'})


class BundleIntegrationTests(unittest.TestCase):
    def test_complete_screened_bundle_retokenizes_and_normalizes_schema(self):
        from transformers import AutoTokenizer
        from experiments.tiny_aya.guard import _snapshot,tokenize_row,_digest as guard_digest
        snapshot,tokenizer_files=_snapshot(train.MODEL,train.REVISION,False)
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        data=bundle()
        for split in ('train','eval_task'):
            data[split]=[tokenize_row(tokenizer,{**row,'instruction':'Hello','response':'Hi'}) for row in data[split]]
        for split in ('alignment','attacks'):
            for row in data[split]:row['instruction']='Hello'
        screening=dict(manifest={},train=data['train'],eval_task=data['eval_task'],config=dict(runtime=dict(tiny_tokenizer_files=tokenizer_files,tiny_template_sha256=guard_digest(tokenizer.chat_template))))
        with tempfile.TemporaryDirectory() as temporary:
            prepared=Path(temporary)/'prepared';screened=Path(temporary)/'screened'
            prepared.mkdir();screened.mkdir()
            (prepared/'manifest.json').write_text('{}');(screened/'manifest.json').write_text('{}')
            for split in ('alignment','attacks'):
                (prepared/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in data[split]))
            with patch.object(train,'verify_bundle',return_value={}),patch.object(train,'load_screening',return_value=screening) as load_screening:
                loaded=train.load_training_bundle(prepared,screened)
                load_screening.assert_called_once_with(screened,prepared)
                self.assertEqual(len(loaded['train']),1800);self.assertEqual(len(loaded['alignment']),72)
                self.assertEqual(loaded['train'][-1]['attention_mask'],[1]*loaded['train'][-1]['sequence_tokens'])
                self.assertEqual(loaded['train'][-1]['input_ids'][-2:],[261001,6])
                data['train'][-1]['labels'][-1]=3
                with self.assertRaises(ValueError):train.load_training_bundle(prepared,screened)


class TinyCohereFixture:

    @classmethod
    def setUpClass(cls):
        threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads,threads)

    def make_model(self):
        from transformers import Cohere2Config,Cohere2ForCausalLM
        from peft import get_peft_model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            base=Cohere2ForCausalLM(Cohere2Config(vocab_size=32,hidden_size=16,intermediate_size=24,
                 num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
                 sliding_window=16,attention_dropout=0.,logit_scale=.2,bos_token_id=1,eos_token_id=3,pad_token_id=0)).to(torch.bfloat16)
            train.pin_attention_backend(base,'eager')
            return get_peft_model(base,train.lora_config())

class Cohere2Tests(TinyCohereFixture,unittest.TestCase):
    def test_selected_suffix_ce_target_coverage_and_zero_reference(self):
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        model=self.make_model()
        coverage=train.verify_adapter(model,expected_layers=2,expected_hidden=16,expected_vocab=32)
        self.assertEqual(coverage['target_count'],14)
        row=dict(input_ids=[1,2,4,5,6],attention_mask=[1]*5,labels=[-100]*3+[5,6],prompt_tokens=3,sequence_tokens=5,supervised_tokens=2)
        parameters=[p for p in model.parameters() if p.requires_grad]
        with train.model_mode(model,training=False,checkpointing=False):
            logits=model(input_ids=torch.tensor([row['input_ids']]),use_cache=False).logits
            expected=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,32),torch.tensor([row['labels']])[:,1:].reshape(-1))
            actual=train.task_loss(model,row,device='cpu',chunk_tokens=1)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        flags=[p.requires_grad for p in model.parameters()]
        diagnostic={}
        value,grads=train.alignment_gradients(model,dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones((1,3),dtype=torch.long)),parameters,safegrad_kl_alignment_loss,require_initial_equivalence=True,diagnostics=diagnostic)
        self.assertEqual(value,0.);self.assertTrue(all(torch.count_nonzero(g)==0 for g in grads))
        self.assertTrue(diagnostic['student_requires_grad']);self.assertEqual(flags,[p.requires_grad for p in model.parameters()])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with torch.no_grad():
            first=next(p for n,p in model.named_parameters() if '.lora_A.' in n);first.fill_(float('inf'))
        with self.assertRaises(ValueError):train.verify_adapter(model,expected_layers=2,expected_hidden=16,expected_vocab=32)

    def test_initialization_and_adapter_bytes_reload_and_tamper(self):
        model=self.make_model()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'init'
            frozen=train.shared_initialization(model,path,create=True)
            self.assertTrue(frozen['zero_B'])
            first=train.adapter_state(model)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:p.add_(1.)
            train.shared_initialization(model,path,create=False)
            self.assertEqual(frozen['tensor_sha256'],train.tensor_digest(train.adapter_state(model)))
            self.assertTrue(all(torch.equal(v,train.adapter_state(model)[k]) for k,v in first.items()))
            with self.assertRaises((ValueError,FileExistsError)):train.shared_initialization(model,path,create=True)
            tensor=path/'adapter_model.safetensors'
            tensor.write_bytes(tensor.read_bytes()+b'bad')
            with self.assertRaises(ValueError):train.shared_initialization(model,path,create=False)


class LedgerTests(unittest.TestCase):
    def records(self,arm='safegrad'):
        plan=train.freeze_training_plan(bundle())
        return [dict(update=u['update'],epoch=u['epoch'],epoch_update=u['epoch_update'],task_ids=u['task_ids'],
                     alignment_ids=u['alignment_ids'] if arm=='safegrad' else [],count=len(u['task_ids']),
                     learning_rate=u['learning_rate'],task_loss=1.,alignment_loss=.1 if arm=='safegrad' else None,
                     gradient_statistics=dict(task_norm=2.,alignment_norm=1.,dot=-1.,cosine=-.5,
                         projection_coefficient=-1.,rho=1.,conflict=True,zero_alignment=False) if arm=='safegrad' else
                         dict(task_norm=2.,alignment_norm=None,dot=None,cosine=None,projection_coefficient=None,rho=0.,conflict=False,zero_alignment=None),
                     preclip_norm=2.,finite_adapter_weights=True,seconds=.1,memory=dict(process_max_rss_gib=.01))
                for u in plan['updates']]

    def test_reconstructs_full_counts_and_rejects_tampered_update_evidence(self):
        plan=train.freeze_training_plan(bundle());records=self.records()
        result=train.validate_ledger(records,plan,arm='safegrad',expected_updates=171)
        self.assertEqual(result['task_exposures'],5400)
        self.assertEqual(result['alignment_exposures'],5400)
        self.assertEqual(result['task_per_language'],dict(en=1800,ko=1800,zh=1800))
        self.assertEqual(set(result['task_counts'].values()),{3})
        self.assertEqual(set(result['alignment_counts'].values()),{75})
        for key,value in [('update',1),('learning_rate',1e-5),('count',32),('task_ids',['other']),
                          ('alignment_ids',[]),('task_loss',float('nan')),('finite_adapter_weights',False),
                          ('gradient_statistics',dict(task_norm=float('inf')))]:
            changed=copy.deepcopy(records);changed[-1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):train.validate_ledger(changed,plan,arm='safegrad',expected_updates=171)
        with self.assertRaises(ValueError):train.validate_ledger(records[:-1],plan,arm='safegrad',expected_updates=171)
        with self.assertRaises(ValueError):train.validate_ledger(records,plan,arm='plain',expected_updates=171)
        partial=train.validate_ledger(records[:1],plan,arm='safegrad',expected_updates=1)
        self.assertEqual(partial['task_exposures'],32)
        self.assertFalse(partial['complete'])

    def test_logit_guard_rejects_without_repair(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError):train.finite_logits_hook(None,None,type('Output',(),dict(logits=torch.tensor([value])))())
        logits=torch.tensor([1.]);output=type('Output',(),dict(logits=logits))()
        self.assertIsNone(train.finite_logits_hook(None,None,output))
        self.assertEqual(logits.item(),1.)


class RunnerTests(TinyCohereFixture,unittest.TestCase):
    def test_feasibility_is_auditable_fresh_reload_and_never_full(self):
        import weakref
        data=bundle()
        for split in ('train','eval_task'):
            for row in data[split]:row.update(input_ids=[1,2,4,5,6],attention_mask=[1]*5,labels=[-100]*3+[5,6],prompt_tokens=3,sequence_tokens=5,supervised_tokens=2)
        for row in data['alignment']:row.update(input_ids=[1,2,3],attention_mask=[1]*3,sequence_tokens=3,prompt_tokens=3)
        data['bindings']={'fixture':'CPU synthetic rows; no model weights or data downloads'}
        models=[]
        def new_model(snapshot,device):
            if models:self.assertIsNone(models[-1](), 'old model must be released before fresh-base reload')
            model=self.make_model();models.append(weakref.ref(model));return model
        verify=train.verify_adapter
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);snapshot=directory/'snapshot';snapshot.mkdir();(snapshot/'config.json').write_text(self.make_model().get_base_model().config.to_json_string())
            checkpoint={'config.json':train._hash_file(snapshot/'config.json')}
            with patch.object(train,'load_training_bundle',return_value=data),patch.object(train,'_snapshot',return_value=(snapshot,checkpoint)),patch.object(train,'_device',return_value='cpu'),patch.object(train,'_new_model',side_effect=new_model),patch.object(train,'verify_adapter',new=lambda model:verify(model,expected_layers=2,expected_hidden=16,expected_vocab=32)),patch.object(train,'expected_adapter_inventory',new=lambda config:train._adapter_inventory(config)):
                output=directory/'safegrad';initialization=directory/'initialization'
                report=train.run_training(directory/'prepared',directory/'screened',output,arm='safegrad',initialization=initialization,create_initialization=True,max_updates=1)
                self.assertEqual(report['status'],'feasibility',report)
                self.assertEqual(report['counts']['task_exposures'],32)
                self.assertTrue(report['reload']['every_adapter_tensor_bitwise_equal'])
                self.assertTrue(report['preflight']['reference_logits_equal'])
                self.assertEqual(report['preflight']['initial_alignment_loss'],0.)
                self.assertEqual(len(models),2)
                with self.assertRaises(ValueError):train.load_training_run(output,directory/'prepared',directory/'screened')
                init_hash=train._hash_file(initialization/'adapter_model.safetensors')
                plain=train.run_training(directory/'prepared',directory/'screened',directory/'plain',arm='plain',initialization=initialization,max_updates=1)
                self.assertEqual(plain['status'],'feasibility',plain)
                self.assertEqual(init_hash,train._hash_file(initialization/'adapter_model.safetensors'))
                self.assertEqual(train._read(output/'run-config.json')['initialization'],train._read(directory/'plain'/'run-config.json')['initialization'])
                self.assertEqual((output/'plan.json').read_bytes(),(directory/'plain'/'plan.json').read_bytes())
                with self.assertRaises((ValueError,FileExistsError)):
                    train.run_training(directory/'prepared',directory/'screened',output,arm='safegrad',initialization=initialization,max_updates=1)
                # Reconstruct a complete artifact fixture from the genuine one-step run;
                # synthetic ledger rows test the validator only, never claim training evidence.
                records=LedgerTests().records()
                with (output/'steps.jsonl').open('w') as handle:
                    for row in records:handle.write(train._json(row)+'\n')
                report.update(status='complete',run_kind='full',optimizer_steps=171,counts=train.validate_ledger(records,train.freeze_training_plan(data),arm='safegrad',expected_updates=171))
                config=train._read(output/'run-config.json');config['max_updates']=None;train._write(output/'run-config.json',config)
                config['platform']['device']='mps';train._write(output/'run-config.json',config)
                report['config_sha256']=train._hash_file(output/'run-config.json')
                report['steps_sha256']=train._hash_file(output/'steps.jsonl')
                train._write(output/'report.json',report)
                valid=train.load_training_run(output,directory/'prepared',directory/'screened')
                self.assertEqual(valid['report']['counts']['task_exposures'],5400)
                # Hash-consistent malformed logits still fail semantic validation.
                from safetensors.torch import save_file,load_file
                probe_path=output/'reload-probe.safetensors';probe_bytes=probe_path.read_bytes()
                actual=load_file(str(probe_path))
                self.assertEqual(actual['before'].shape,(1,1,32))
                self.assertEqual(actual['before'].dtype,torch.bfloat16)
                for shape,dtype in (((1,),torch.float32),((1,),torch.bfloat16),
                                    ((2,1,32),torch.bfloat16),((1,2,32),torch.bfloat16),
                                    ((1,1,31),torch.bfloat16),((1,1,32),torch.float32)):
                    with self.subTest(probe_shape=shape,probe_dtype=dtype):
                        save_file({key:torch.zeros(shape,dtype=dtype) for key in ('before','after')},str(probe_path))
                        altered=copy.deepcopy(report)
                        altered['reload']['probe_sha256']=train._hash_file(probe_path)
                        altered['reload']['max_abs_logit_difference']=0.
                        train._write(output/'report.json',altered)
                        with self.assertRaisesRegex(ValueError,'Invalid reload logits'):
                            train.load_training_run(output,directory/'prepared',directory/'screened')
                # Exercise the real pinned vocabulary width without allocating
                # pretrained weights: this is still a synthetic consumer fixture.
                snapshot_config=snapshot/'config.json';snapshot_bytes=snapshot_config.read_bytes()
                pinned_width=train._read(snapshot_config);pinned_width['vocab_size']=262144
                train._write(snapshot_config,pinned_width);checkpoint['config.json']=train._hash_file(snapshot_config)
                wide_config=copy.deepcopy(config);wide_config['checkpoint_files']=copy.deepcopy(checkpoint)
                train._write(output/'run-config.json',wide_config)
                wide_report=copy.deepcopy(report);wide_report['config_sha256']=train._hash_file(output/'run-config.json')
                for dtype,accept in ((torch.bfloat16,True),(torch.float32,False)):
                    save_file({key:torch.zeros((1,1,262144),dtype=dtype) for key in ('before','after')},str(probe_path))
                    wide_report['reload']['probe_sha256']=train._hash_file(probe_path)
                    wide_report['reload']['max_abs_logit_difference']=0.;train._write(output/'report.json',wide_report)
                    if accept:
                        self.assertEqual(train.load_training_run(output,directory/'prepared',directory/'screened')['counts']['updates'],171)
                    else:
                        with self.assertRaisesRegex(ValueError,'Invalid reload logits'):
                            train.load_training_run(output,directory/'prepared',directory/'screened')
                snapshot_config.write_bytes(snapshot_bytes);checkpoint['config.json']=train._hash_file(snapshot_config)
                train._write(output/'run-config.json',config)
                probe_path.write_bytes(probe_bytes);train._write(output/'report.json',report)
                self.assertEqual(train.load_training_run(output,directory/'prepared',directory/'screened')['counts']['updates'],171)
                for name in ('plan.json','run-config.json','steps.jsonl','sources/train.py','adapter/adapter_model.safetensors','adapter/adapter_config.json'):
                    path=output/name;original=path.read_bytes();path.write_bytes(original+b'bad')
                    with self.subTest(name=name),self.assertRaises((ValueError,json.JSONDecodeError)):
                        train.load_training_run(output,directory/'prepared',directory/'screened')
                    path.write_bytes(original)
                # Rehashed semantic mutations must still be rejected, not just byte changes.
                for key,value in [('settings',{**train.SETTINGS,'accumulation':8}),('arm','plain'),('data',{'wrong':'data'}),('max_updates',1)]:
                    changed=copy.deepcopy(config);changed[key]=value;train._write(output/'run-config.json',changed)
                    altered=copy.deepcopy(report);altered['config_sha256']=train._hash_file(output/'run-config.json');train._write(output/'report.json',altered)
                    with self.subTest(config=key),self.assertRaises(ValueError):train.load_training_run(output,directory/'prepared',directory/'screened')
                train._write(output/'run-config.json',config)
                report['status']='running';train._write(output/'report.json',report)
                with self.assertRaises(ValueError):train.load_training_run(output,directory/'prepared',directory/'screened')


    def test_runner_projects_group_means_once(self):
        data=bundle()
        for split in ('train','eval_task'):
            for row in data[split]:row.update(input_ids=[1,2,4,5,6],attention_mask=[1]*5,labels=[-100]*3+[5,6],prompt_tokens=3,sequence_tokens=5,supervised_tokens=2)
        for row in data['alignment']:row.update(input_ids=[1,2,3],attention_mask=[1]*3,sequence_tokens=3,prompt_tokens=3)
        data['bindings']={'fixture':'constructed projection gradients'}
        order=train.freeze_training_plan(data)['updates'][0]['task_ids'];indices={identity:i for i,identity in enumerate(order)}
        actual_alignment=train.alignment_gradients;align_calls=0
        def task_objective(model,row,**kwargs):
            parameters=[p for p in model.parameters() if p.requires_grad]
            first=parameters[0].flatten();index=indices.get(row['id'],0)
            x=-3. if index%2==0 else 1.
            return sum(p.sum()*0. for p in parameters)+first[0]*x+first[1]+100.
        def alignment_objective(model,batch,parameters,helper,**kwargs):
            nonlocal align_calls
            if kwargs.get('require_initial_equivalence'):return actual_alignment(model,batch,parameters,helper,**kwargs)
            gradients=[torch.zeros_like(p) for p in parameters]
            gradients[0].flatten()[:2]=torch.tensor([1.,0.] if align_calls%2==0 else [-1.,1.])
            align_calls+=1
            return .1,gradients
        verify=train.verify_adapter
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);snapshot=directory/'snapshot';snapshot.mkdir();(snapshot/'config.json').write_text('{}')
            checkpoint={'config.json':train._hash_file(snapshot/'config.json')}
            with patch.object(train,'load_training_bundle',return_value=data),patch.object(train,'_snapshot',return_value=(snapshot,checkpoint)),patch.object(train,'_device',return_value='cpu'),patch.object(train,'_new_model',new=lambda *_:self.make_model()),patch.object(train,'verify_adapter',new=lambda model:verify(model,expected_layers=2,expected_hidden=16,expected_vocab=32)),patch.object(train,'task_loss',new=task_objective),patch.object(train,'alignment_gradients',new=alignment_objective):
                output=directory/'run'
                report=train.run_training(directory/'prepared',directory/'screened',output,arm='safegrad',initialization=directory/'init',create_initialization=True,max_updates=1)
            self.assertEqual(report['status'],'feasibility',report)
            record=train._rows(output/'steps.jsonl')[0]
            self.assertEqual(record['gradient_statistics']['dot'],.5)
            self.assertEqual(record['gradient_statistics']['projection_coefficient'],0.)
            initial=train._load_initialization_files(output/'initialization',train._read(output/'initialization'/'manifest.json'))
            final=train._load_adapter_files(output/'adapter',report['adapter'])
            first_name='base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
            # Mean projection gives a positive first-coordinate gradient; the
            # required projection of means gives -1 and increases this weight.
            self.assertGreater(final[first_name].flatten()[0],initial[first_name].flatten()[0])

    def test_runtime_failure_leaves_audit_and_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'failed'
            with patch.object(train,'load_training_bundle',side_effect=ValueError('bad data')):
                report=train.run_training(Path('prepared'),Path('screened'),output,arm='plain',initialization=Path(temporary)/'init')
            self.assertEqual(report['status'],'failed');self.assertEqual(report['optimizer_steps'],0)
            self.assertIn('bad data',(output/'error.log').read_text())
            self.assertFalse((output/'.writer.lock').exists())
            with self.assertRaises(FileExistsError):
                train.run_training(Path('prepared'),Path('screened'),output,arm='plain',initialization=Path(temporary)/'init')


class AdditionalBoundaryTests(unittest.TestCase):
    def test_checkpoint_shapes_are_independent_of_saved_adapter_manifest(self):
        config=dict(model_type='cohere2',num_hidden_layers=36,hidden_size=2048,vocab_size=262144,
                    num_attention_heads=16,num_key_value_heads=4,head_dim=128,intermediate_size=11008)
        expected=train.expected_adapter_inventory(config)
        self.assertEqual(len(expected),504)
        self.assertEqual(expected['base_model.model.model.layers.35.self_attn.k_proj.lora_B.weight'],dict(shape=[512,4],dtype='torch.float32'))
        self.assertEqual(expected['base_model.model.model.layers.0.mlp.down_proj.lora_A.weight'],dict(shape=[4,11008],dtype='torch.float32'))
        for key,value in [('num_hidden_layers',35),('model_type','cohere'),('hidden_size',16),('vocab_size',100)]:
            with self.assertRaises(ValueError):train.expected_adapter_inventory({**config,key:value})

    def test_gradient_shapes_and_overflow_are_rejected(self):
        for gradients in (([torch.ones(2)],[torch.ones(3)]),([torch.ones(1)],[torch.ones(1),torch.ones(1)]),
                          ([torch.full((1,),3e38)],[torch.full((1,),3e38)])):
            with self.assertRaises(ValueError):train.accumulate_gradient_mean(iter(gradients),count=2)

    def test_task_and_alignment_token_caps_reject_overlong_inputs(self):
        class Tokenizer:
            eos_token_id=6
            def __len__(self):return 262144
            def convert_tokens_to_ids(self,token):return 261001
            def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,return_dict,truncation):
                self.kwargs=(tokenize,return_dict,truncation)
                return [2]*1025 if add_generation_prompt else [2]*1025+[9,261001,6]
        tokenizer=Tokenizer()
        with self.assertRaises(ValueError):train.tokenize_alignment(tokenizer,dict(lang='en',instruction='long'))
        with self.assertRaises(ValueError):train.verify_task_row(tokenizer,dict(lang='en',instruction='long',response='x'))
        self.assertEqual(tokenizer.kwargs,(True,False,False))


if __name__=='__main__':unittest.main()
