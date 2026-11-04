"""Pinned Tiny Aya adapter training with equal-example accumulated SafeGrad.

Local-only model loading; fresh outputs and shared immutable initialization.
A feasibility prefix never qualifies as a completed training run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import time
import traceback

from experiments.multilingual_repair.train import (
    combine_gradients, model_mode, disabled_adapter_reference, chunked_alignment_loss,
    alignment_gradients, task_loss, pin_attention_backend, verify_attention_backend,
)
from experiments.tiny_aya.prepare import CONFIG, verify_bundle
from experiments.tiny_aya.guard import load_screening, tokenize_row, _snapshot, _digest as screening_digest

MODEL = CONFIG['model']
REVISION = CONFIG['model_revision']
LANGUAGES = ('en', 'ko', 'zh')
TARGETS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
SETTINGS = dict(model=MODEL, revision=REVISION, layers=36, hidden_size=2048, vocab_size=262144,
    device='mps', base_dtype='bfloat16', adapter_dtype='float32', attention='eager',
    rank=4, alpha=8, dropout=0., bias='none', targets=sorted(TARGETS), epochs=3,
    seed=42, alignment_seed=43, accumulation=32, updates=171, weight_decay=0.,
    betas=[.9,.999], epsilon=1e-8, foreach=False, max_grad_norm=1., rho=1.,
    mps_memory_fraction=.8, gradient_checkpointing_use_reentrant=False,
    learning_rate='1.2e-6+0.5*(2.5e-5-1.2e-6)*(1+cos(pi*k/170))',
    task_objective='mean suffix CE per example; equal-example mean over actual update group',
    alignment_objective='KL(reference||student), temperature1, batchmean prompt-position sum; equal-example group mean',
    projection='one global projection after task/alignment gradient means')


def require(condition, message):
    if not condition: raise ValueError(message)


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('w',encoding='utf-8') as handle:
        handle.write(_json(value)+'\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def _rows(path):
    blob=Path(path).read_bytes()
    require(not blob or blob.endswith(b'\n'),'Incomplete JSONL terminal newline')
    return [json.loads(line) for line in blob.decode().splitlines()]


def _families(rows,count,languages):
    grouped=defaultdict(list);seen=set()
    for row in rows:
        require(isinstance(row,dict) and row.get('lang') in languages,'Invalid optimizer/partition language')
        identity=row.get('id');family=row.get('family_id')
        require(isinstance(identity,str) and isinstance(family,str) and family and identity==f'{family}:{row["lang"]}'
                and identity not in seen,'Invalid/duplicate row identity')
        seen.add(identity);grouped[family].append(row['lang'])
    require(len(grouped)==count and len(rows)==count*len(languages)
            and all(sorted(langs)==sorted(languages) for langs in grouped.values()),'Incorrect complete family coverage')
    return set(grouped)


def _validate_partitions(bundle):
    train=_families(bundle['train'],600,LANGUAGES)
    evaluation=_families(bundle['eval_task'],100,LANGUAGES)
    align=_families(bundle['alignment'],24,LANGUAGES)
    attack=_families(bundle['attacks'],100,(*LANGUAGES,'sw'))
    require(not train&evaluation,'Train/evaluation families overlap')
    require(not align&attack,'Alignment/attack families overlap')
    require(not (train|evaluation)&(align|attack),'Task/attack source families overlap')


def freeze_training_plan(bundle:dict)->dict:
    """Pure CPU: freeze all three epochs and the continuous alignment cycle."""
    _validate_partitions(bundle)
    alignment=[r['id'] for r in bundle['alignment']];random.Random(43).shuffle(alignment)
    updates=[];exposure=0;orders=[]
    for epoch in range(3):
        order=[r['id'] for r in bundle['train']];random.Random(42+epoch).shuffle(order);orders.append(order)
        for start in range(0,len(order),32):
            ids=order[start:start+32];k=len(updates)
            updates.append(dict(update=k,epoch=epoch,epoch_update=start//32,task_ids=ids,
                alignment_ids=[alignment[(exposure+i)%72] for i in range(len(ids))],
                learning_rate=1.2e-6+.5*(2.5e-5-1.2e-6)*(1+math.cos(math.pi*k/170))))
            exposure+=len(ids)
    plan=dict(schema_version=1,settings=copy.deepcopy(SETTINGS),epoch_orders=orders,alignment_cycle=alignment,updates=updates,
              expected=dict(updates=171,task_exposures=5400,uses_per_task=3,alignment_exposures=5400,
                            uses_per_alignment=75,task_per_language={'en':1800,'ko':1800,'zh':1800}))
    plan['sha256']=_digest(plan)
    return plan


def select_updates(plan,max_updates):
    require(max_updates is None or type(max_updates) is int and 1<=max_updates<171,'Feasibility max_updates must be 1..170')
    return plan['updates'][:max_updates] if max_updates is not None else plan['updates']


class GradientMean:
    """Streaming FP32 sum: retains only one buffer per parameter, never graphs."""
    def __init__(self):self.buffers=None;self.count=0
    def add(self,gradients):
        import torch
        require(isinstance(gradients,(tuple,list)) and bool(gradients),'Missing gradient coverage')
        require(all(isinstance(g,torch.Tensor) and torch.isfinite(g).all().item() for g in gradients),'Missing/nonfinite gradient')
        if self.buffers is None:self.buffers=[g.detach().to(dtype=torch.float32).clone() for g in gradients]
        else:
            require(len(gradients)==len(self.buffers),'Gradient coverage mismatch')
            for buffer,g in zip(self.buffers,gradients):
                require(buffer.shape==g.shape and buffer.device==g.device,'Gradient shape/device mismatch')
                buffer.add_(g.detach().float())
                require(torch.isfinite(buffer).all().item(),'Nonfinite accumulated gradient')
        self.count+=1
    def mean(self,count):
        require(type(count) is int and count>0 and count==self.count,'Gradient count mismatch')
        return [g/count for g in self.buffers]


def accumulate_gradient_mean(gradient_rows,*,count):
    accumulator=GradientMean()
    for row in gradient_rows:accumulator.add(row)
    return accumulator.mean(count)


def finalize_gradients(task,alignment=None):
    """Inputs are group means; perform exactly one global surgery here."""
    import torch
    if alignment is not None:return combine_gradients(task,alignment,rho=1.)
    require(bool(task) and all(isinstance(g,torch.Tensor) and torch.isfinite(g).all().item() for g in task),'Invalid task gradient')
    norm=math.sqrt(math.fsum(g.float().square().sum().item() for g in task))
    require(math.isfinite(norm),'Nonfinite task norm')
    return task,dict(task_norm=norm,alignment_norm=None,dot=None,cosine=None,projection_coefficient=None,rho=0.,conflict=False,zero_alignment=None)


def verify_task_row(tokenizer,row):
    require(row.get('lang') in LANGUAGES,'Invalid optimizer language')
    rebuilt=tokenize_row(tokenizer,row)
    for key in ('input_ids','labels','prompt_tokens','sequence_tokens','supervised_tokens'):
        require(type(row.get(key)) is type(rebuilt[key]) and row[key]==rebuilt[key],f'Task token mismatch: {key}')
    require(all(type(i) is int and 0<=i<len(tokenizer) for i in row['input_ids']),'Invalid task token IDs')
    require(all(type(i) is int for i in row['labels']),'Invalid task label types')
    if 'attention_mask' in row:
        require(row['attention_mask']==[1]*len(row['input_ids']) and all(type(i) is int for i in row['attention_mask']),'Invalid stored attention mask')
    return {**row,'attention_mask':[1]*len(row['input_ids'])}


def tokenize_alignment(tokenizer,row):
    require(row.get('lang') in LANGUAGES and 'response' not in row,'Invalid prompt-only alignment row')
    ids=list(tokenizer.apply_chat_template([dict(role='user',content=row['instruction'])],tokenize=True,
              add_generation_prompt=True,return_dict=False,truncation=False))
    require(0<len(ids)<=1024 and all(type(i) is int and 0<=i<len(tokenizer) for i in ids),'Alignment token budget/IDs invalid')
    return {**row,'input_ids':ids,'attention_mask':[1]*len(ids),'sequence_tokens':len(ids),'prompt_tokens':len(ids)}


def load_training_bundle(prepared:Path,screened:Path)->dict:
    """Verify both complete source artifacts, partitions and exact official tokens."""
    from transformers import AutoTokenizer
    prepared,screened=Path(prepared),Path(screened)
    manifest=verify_bundle(prepared)
    screening=load_screening(screened,prepared)
    snapshot,files=_snapshot(MODEL,REVISION,False)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    runtime=screening['config']['runtime']
    require(runtime['tiny_tokenizer_files']==files and runtime['tiny_template_sha256']==screening_digest(tokenizer.chat_template),'Screened Tiny tokenizer/template binding mismatch')
    bundle=dict(train=screening['train'],eval_task=screening['eval_task'],
                alignment=_rows(prepared/'alignment.jsonl'),attacks=_rows(prepared/'attacks.jsonl'))
    _validate_partitions(bundle)
    for split in ('train','eval_task'):bundle[split]=[verify_task_row(tokenizer,r) for r in bundle[split]]
    bundle['alignment']=[tokenize_alignment(tokenizer,r) for r in bundle['alignment']]
    bundle['bindings']=dict(prepared_manifest_sha256=_hash_file(prepared/'manifest.json'),
        screened_manifest_sha256=_hash_file(screened/'manifest.json'),prepared_manifest=manifest,
        screened_manifest=screening['manifest'],tokenizer_files=files,template_sha256=screening_digest(tokenizer.chat_template),effective_template_sha256=_digest(tokenizer.get_chat_template()),
        data_sha256=_digest({s:bundle[s] for s in ('train','eval_task','alignment','attacks')}))
    return bundle


def lora_config():
    from peft import LoraConfig
    return LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,bias='none',target_modules=list(TARGETS),task_type='CAUSAL_LM')


def verify_adapter(model,*,expected_layers=36,expected_hidden=2048,expected_vocab=262144):
    import torch
    base=model.get_base_model()
    require(base.config.model_type=='cohere2' and base.config.num_hidden_layers==expected_layers
            and base.config.hidden_size==expected_hidden and base.config.vocab_size==expected_vocab,'Unexpected Cohere2 architecture')
    require(base.config.attention_dropout==0. and all(module.p==0. for module in model.modules() if isinstance(module,torch.nn.Dropout)),'Training requires zero dropout')
    attention=verify_attention_backend(model,'eager')
    require(attention['attention_modules']==expected_layers,'Attention module coverage mismatch')
    targets=defaultdict(set)
    for name,module in model.named_modules():
        if hasattr(module,'lora_A'):
            parent,_,target=name.rpartition('.')
            require(target in TARGETS and 'default' in module.lora_A and 'default' in module.lora_B,'Unexpected LoRA module')
            layer=parent.split('.layers.')[-1].split('.')[0]
            require(layer.isdigit(),'Unrecognized LoRA block path')
            require(target not in targets[int(layer)],'Duplicate LoRA target')
            targets[int(layer)].add(target)
    require(set(targets)==set(range(expected_layers)) and all(t==set(TARGETS) for t in targets.values()),'LoRA target coverage mismatch')
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    require(bool(trainable) and all(('.lora_A.default.weight' in n or '.lora_B.default.weight' in n)
            and p.dtype==torch.float32 and torch.isfinite(p).all().item() for n,p in trainable.items()),'Invalid trainable adapter parameters')
    require(len(trainable)==expected_layers*14,'Adapter parameter coverage mismatch')
    require(all(p.grad is None and p.dtype==torch.bfloat16 for n,p in model.named_parameters() if n not in trainable),'Frozen base dtype/gradient mismatch')
    config=model.peft_config['default']
    require(config.r==4 and config.lora_alpha==8 and config.lora_dropout==0. and config.bias=='none'
            and set(config.target_modules)==set(TARGETS) and not config.use_dora and not config.use_rslora
            and not config.modules_to_save and not config.rank_pattern and not config.alpha_pattern,'Unapproved LoRA configuration')
    return dict(target_count=expected_layers*7,attention=attention,trainable_tensors=len(trainable),
                trainable_parameters=sum(p.numel() for p in trainable.values()))


def _adapter_inventory(config):
    hidden=config['hidden_size'];intermediate=config['intermediate_size']
    head_dim=config.get('head_dim') or hidden//config['num_attention_heads']
    query=config['num_attention_heads']*head_dim;kv=config['num_key_value_heads']*head_dim
    shapes={'q_proj':(query,hidden),'k_proj':(kv,hidden),'v_proj':(kv,hidden),
            'o_proj':(hidden,query),'gate_proj':(intermediate,hidden),'up_proj':(intermediate,hidden),
            'down_proj':(hidden,intermediate)}
    expected={}
    for layer in range(config['num_hidden_layers']):
        for target,(out_size,in_size) in shapes.items():
            group='self_attn' if target in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'
            prefix=f'base_model.model.model.layers.{layer}.{group}.{target}'
            expected[prefix+'.lora_A.weight']=dict(shape=[4,in_size],dtype='torch.float32')
            expected[prefix+'.lora_B.weight']=dict(shape=[out_size,4],dtype='torch.float32')
    return expected


def expected_adapter_inventory(config):
    require(config.get('model_type')=='cohere2' and config.get('num_hidden_layers')==36
            and config.get('hidden_size')==2048 and config.get('vocab_size')==262144,'Pinned checkpoint architecture mismatch')
    return _adapter_inventory(config)


def adapter_state(model):
    import torch
    from peft import get_peft_model_state_dict
    state={k:v.detach().cpu().contiguous().clone() for k,v in get_peft_model_state_dict(model,save_embedding_layers=False).items()}
    require(bool(state) and all(('lora_A' in k or 'lora_B' in k) and v.dtype==torch.float32
            and torch.isfinite(v).all().item() for k,v in state.items()),'Invalid adapter-only state')
    return state


def tensor_digest(state):
    """Ordered names/shapes/dtypes plus exact tensor bytes (independent of file layout)."""
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update((_json(dict(name=name,shape=list(value.shape),dtype=str(value.dtype)))+'\n').encode())
        digest.update(value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def _save_adapter(model,path):
    from safetensors.torch import save_file
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    state=adapter_state(model)
    model.peft_config['default'].base_model_name_or_path=MODEL
    model.peft_config['default'].revision=REVISION
    model.peft_config['default'].save_pretrained(path)
    save_file(state,str(path/'adapter_model.safetensors'),metadata={'format':'pt'})
    return dict(files={p.name:_hash_file(p) for p in sorted(path.iterdir())},tensor_sha256=tensor_digest(state),
                tensor_inventory={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in sorted(state.items())})


def _load_adapter_files(path,manifest):
    import torch
    from safetensors.torch import load_file
    path=Path(path)
    require(set(p.name for p in path.iterdir())=={'adapter_config.json','adapter_model.safetensors'},'Unexpected adapter inventory')
    require(manifest['files']=={p.name:_hash_file(p) for p in sorted(path.iterdir())},'Adapter file hash mismatch')
    config=_read(path/'adapter_config.json')
    # Full PEFT defaults are frozen in the file digest; reject all added features independently.
    from peft import LoraConfig
    expected=LoraConfig.from_pretrained(path)
    baseline=lora_config();baseline.base_model_name_or_path=MODEL;baseline.revision=REVISION
    # Save canonical PEFT JSON serialization is stable for sets/enums through to_dict conversion.
    def canonical(c):return json.loads(json.dumps(c.to_dict(),default=lambda x:sorted(x) if isinstance(x,set) else x.value))
    require(canonical(expected)==canonical(baseline),'Unapproved adapter configuration')
    require(config.get('base_model_name_or_path')==MODEL and config.get('revision')==REVISION,'Adapter base pin mismatch')
    state=load_file(str(path/'adapter_model.safetensors'),device='cpu')
    require(bool(state) and all(('lora_A' in k or 'lora_B' in k) and v.dtype==torch.float32
        and torch.isfinite(v).all().item() for k,v in state.items()),'Invalid saved adapter tensors')
    require(tensor_digest(state)==manifest['tensor_sha256'],'Adapter tensor digest mismatch')
    require({k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in sorted(state.items())}==manifest['tensor_inventory'],'Adapter tensor inventory mismatch')
    return state


def _put_state(model,state):
    import torch
    from peft import set_peft_model_state_dict
    original=adapter_state(model)
    require(original.keys()==state.keys() and all(original[k].shape==v.shape and original[k].dtype==v.dtype for k,v in state.items()),'Adapter name/shape/dtype mismatch')
    set_peft_model_state_dict(model,state)
    actual=adapter_state(model)
    require(all(torch.equal(v,actual[k]) for k,v in state.items()),'Adapter load not bitwise exact')


def shared_initialization(model,path,*,create):
    """One fresh adapter-only initialization, never overwritten by either arm."""
    import torch
    path=Path(path)
    if create:
        require(not path.exists(),'Initialization already exists')
        state=adapter_state(model)
        require(any('lora_B' in k for k in state) and all(torch.count_nonzero(v).item()==0 for k,v in state.items() if 'lora_B' in k),'Initialization B must be exactly zero')
        manifest=_save_adapter(model,path);manifest.update(schema_version=1,zero_B=True,model=MODEL,revision=REVISION)
        _write(path/'manifest.json',manifest)
    else:
        require(path.is_dir(),'Shared initialization missing')
        manifest=_read(path/'manifest.json')
    require(manifest.get('schema_version')==1 and manifest.get('zero_B') is True and manifest.get('model')==MODEL and manifest.get('revision')==REVISION,'Invalid initialization manifest')
    require(set(p.name for p in path.iterdir())=={'manifest.json','adapter_config.json','adapter_model.safetensors'},'Initialization file inventory mismatch')
    # Adapter loader checks its exact inventory; initialization has one extra manifest.
    state=_load_initialization_files(path,manifest)
    require(all(torch.count_nonzero(v).item()==0 for k,v in state.items() if 'lora_B' in k),'Shared initialization B changed')
    _put_state(model,state)
    return manifest


def _load_initialization_files(path,manifest):
    """Validate init without mutating or copying its files."""
    import torch
    from safetensors.torch import load_file
    require(manifest['files']=={name:_hash_file(path/name) for name in ('adapter_config.json','adapter_model.safetensors')},'Initialization file hash mismatch')
    state=load_file(str(path/'adapter_model.safetensors'),device='cpu')
    require(tensor_digest(state)==manifest['tensor_sha256'],'Initialization tensor digest mismatch')
    require({k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in sorted(state.items())}==manifest['tensor_inventory'],'Initialization inventory mismatch')
    require(bool(state) and all(('lora_A' in k or 'lora_B' in k) and v.dtype==torch.float32 and torch.isfinite(v).all().item() for k,v in state.items()),'Invalid initialization state')
    from peft import LoraConfig
    config=LoraConfig.from_pretrained(path);baseline=lora_config();baseline.base_model_name_or_path=MODEL;baseline.revision=REVISION
    require(config.to_dict()==baseline.to_dict(),'Initialization PEFT configuration mismatch')
    return state


def _finite_number(value,*,nonnegative=False):
    return type(value) in (float,int) and math.isfinite(value) and (not nonnegative or value>=0)


def finite_logits_hook(_module,_arguments,output):
    import torch
    logits=output.logits
    require(isinstance(logits,torch.Tensor) and torch.isfinite(logits).all().item(),'Nonfinite model logits')


def validate_ledger(records,plan,*,arm,expected_updates):
    """Reconstruct exact ordered exposures and finite step evidence independently."""
    require(arm in ('plain','safegrad'),'Invalid arm')
    require(type(expected_updates) is int and 1<=expected_updates<=171 and len(records)==expected_updates,'Incomplete update ledger')
    task=Counter();align=Counter();languages=Counter()
    for record,expected in zip(records,plan['updates']):
        for key in ('update','epoch','epoch_update','task_ids','learning_rate'):
            require(type(record.get(key)) is type(expected[key]) and record[key]==expected[key],f'Ledger {key} mismatch')
        ids=expected['task_ids'];alignment_ids=expected['alignment_ids'] if arm=='safegrad' else []
        require(record.get('alignment_ids')==alignment_ids and type(record.get('count')) is int and record['count']==len(ids),'Ledger alignment/count mismatch')
        require(_finite_number(record.get('task_loss'),nonnegative=True),'Invalid task loss')
        require(_finite_number(record.get('alignment_loss')) if arm=='safegrad' else record.get('alignment_loss') is None,'Invalid alignment loss')
        # Floating point KL may have a finite negative roundoff residual: never silently clamp it.
        require(_finite_number(record.get('preclip_norm'),nonnegative=True) and _finite_number(record.get('seconds'),nonnegative=True)
                and record.get('finite_adapter_weights') is True,'Invalid optimizer evidence')
        stats=record.get('gradient_statistics',{})
        require(_finite_number(stats.get('task_norm'),nonnegative=True),'Invalid task gradient statistic')
        if arm=='safegrad':
            require(all(_finite_number(stats.get(k)) for k in ('alignment_norm','dot','projection_coefficient','rho'))
                    and stats['alignment_norm']>=0 and stats['rho']==1.
                    and (stats.get('cosine') is None or _finite_number(stats['cosine']))
                    and type(stats.get('conflict')) is bool and type(stats.get('zero_alignment')) is bool,'Invalid projection statistics')
            sq=stats['alignment_norm']**2;dot=stats['dot']
            conflict=dot<0 and sq>0
            require(stats['conflict']==conflict and stats['zero_alignment']==(sq==0)
                    and math.isclose(stats['projection_coefficient'],dot/sq if conflict else 0.,rel_tol=1e-12,abs_tol=1e-15),'Inconsistent global projection evidence')
            cosine=dot/(stats['task_norm']*stats['alignment_norm']) if stats['task_norm']>0 and sq>0 else None
            require(stats['cosine'] is None if cosine is None else stats['cosine'] is not None and math.isclose(stats['cosine'],cosine,rel_tol=1e-12,abs_tol=1e-15),'Inconsistent cosine evidence')
        else:
            require(stats.get('rho')==0. and all(stats.get(k) is None for k in ('alignment_norm','dot','cosine','projection_coefficient','zero_alignment'))
                    and stats.get('conflict') is False,'Plain arm has alignment statistics')
        memory=record.get('memory')
        require(isinstance(memory,dict) and bool(memory) and all(_finite_number(v,nonnegative=True) for v in memory.values()),'Invalid memory statistics')
        task.update(ids);align.update(alignment_ids);languages.update(i.rsplit(':',1)[1] for i in ids)
    complete=expected_updates==171
    if complete:
        require(set(task.values())=={3} and len(task)==1800 and sum(task.values())==5400
                and dict(languages)==dict(en=1800,ko=1800,zh=1800),'Incomplete task exposures')
        require((len(align)==72 and set(align.values())=={75}) if arm=='safegrad' else not align,'Incomplete alignment exposures')
    return dict(complete=complete,updates=len(records),task_exposures=sum(task.values()),alignment_exposures=sum(align.values()),
                task_per_language=dict(sorted(languages.items())),task_counts=dict(sorted(task.items())),alignment_counts=dict(sorted(align.items())))


def _device():
    import torch
    require(torch.backends.mps.is_available(),'Pinned MPS device unavailable')
    torch.mps.set_per_process_memory_fraction(.8)
    return 'mps'


def _new_model(snapshot,device):
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model
    import torch
    base=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
               dtype=torch.bfloat16,attn_implementation='eager').to(device)
    pin_attention_backend(base,'eager')
    model=get_peft_model(base,lora_config())
    require(all(p.device.type==device for p in model.parameters()),'Unexpected model device')
    verify_adapter(model)
    return model


def _reset_seed(device):
    import torch
    random.seed(42);torch.manual_seed(42)
    if device=='mps':torch.mps.manual_seed(42)


def _memory(device):
    import torch
    record=dict(process_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if platform.system()=='Darwin' else 2**20))
    if device=='mps':
        torch.mps.synchronize()
        record.update(mps_tensor_gib=torch.mps.current_allocated_memory()/2**30,mps_driver_gib=torch.mps.driver_allocated_memory()/2**30)
    return record


def _clear_device(device):
    import torch
    gc.collect()
    if device=='mps':torch.mps.empty_cache()


def _snapshot_sources(output):
    """Archive exact implementation bytes: later validators verify these snapshots.

    Validation does not assume the user's current checkout still equals a historic
    run. Frozen snapshot hashes bind the evidence without blocking later analysis.
    """
    import inspect
    from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
    from safetune.runner.utils.model_utils import load_model,lora_wrap
    files={'train.py':Path(__file__),'numerical-helpers.py':Path(inspect.getfile(task_loss)),
           'guard.py':Path(inspect.getfile(load_screening)),'prepare.py':Path(inspect.getfile(verify_bundle)),
           'tiny-config.json':Path(__file__).with_name('config.json'),
           'safetune-safegrad.py':Path(inspect.getfile(safegrad_kl_alignment_loss)),
           'safetune-model-utils.py':Path(inspect.getfile(load_model))}
    directory=output/'sources';directory.mkdir()
    for name,path in files.items():(directory/name).write_bytes(path.read_bytes())
    return {name:_hash_file(directory/name) for name in sorted(files)}


@contextmanager
def _fresh_output(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    lock=output/'.writer.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        with os.fdopen(descriptor,'w') as handle:
            handle.write(_json(dict(pid=os.getpid(),created_unix=time.time()))+'\n');handle.flush();os.fsync(handle.fileno())
        yield
    finally:lock.unlink()


def _probe(model,row,device):
    import torch
    with model_mode(model,training=False,checkpointing=False),torch.no_grad():
        ids=torch.tensor([row['input_ids'][:row['prompt_tokens']]],dtype=torch.long,device=device)
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),logits_to_keep=1,use_cache=False).logits
    require(torch.isfinite(logits).all().item(),'Nonfinite reload probe logits')
    return logits.detach().cpu().contiguous()


def run_training(prepared:Path,screened:Path,output:Path,*,arm:str,initialization:Path,
                 create_initialization:bool=False,max_updates:int|None=None)->dict:
    """Run from zero initialization once; failures remain auditable, never resumed."""
    require(arm in ('plain','safegrad'),'Unknown training arm')
    require(type(create_initialization) is bool,'Invalid create_initialization')
    # Validate feasibility before making a directory or loading anything.
    require(max_updates is None or type(max_updates) is int and 1<=max_updates<171,'Feasibility max_updates must be 1..170')
    output=Path(output);initialization=Path(initialization)
    require(output.resolve()!=initialization.resolve() and not output.resolve().is_relative_to(initialization.resolve())
            and not initialization.resolve().is_relative_to(output.resolve()),'Initialization/output directories must be independent')
    start=time.perf_counter()
    report=dict(schema_version=1,status='running',stage='verify_data',arm=arm,optimizer_steps=0,
                run_kind='feasibility' if max_updates is not None else 'full',
                memory_scope='Synchronized MPS samples per optimizer update and process RSS high-water; not a continuous allocator peak.',
                scope='Local LoRA training diagnostics; no safety or quality improvement claim.')
    with _fresh_output(output):
        _write(output/'report.json',report)
        model=None;hook=None
        try:
            from experiments.multilingual_repair.train import verify_environment
            from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
            from safetensors.torch import save_file
            import torch
            bundle=load_training_bundle(Path(prepared),Path(screened))
            plan=freeze_training_plan(bundle);updates=select_updates(plan,max_updates)
            _write(output/'plan.json',plan)
            snapshot,checkpoint_files=_snapshot(MODEL,REVISION,True)
            environment=verify_environment()
            sources=_snapshot_sources(output)
            device=_device();_reset_seed(device)
            report['stage']='load_base';_write(output/'report.json',report)
            model=_new_model(snapshot,device)
            coverage=verify_adapter(model)
            init=shared_initialization(model,initialization,create=create_initialization)
            init_manifest_sha256=_hash_file(initialization/'manifest.json')
            # Copy immutable initialization so evaluator never depends on an external path.
            (output/'initialization').mkdir()
            for name in ('manifest.json','adapter_config.json','adapter_model.safetensors'):
                (output/'initialization'/name).write_bytes((initialization/name).read_bytes())
            config=dict(schema_version=1,settings=SETTINGS,arm=arm,max_updates=max_updates,
                data=bundle['bindings'],checkpoint_files=checkpoint_files,environment=environment,
                platform=dict(system=platform.system(),release=platform.release(),machine=platform.machine(),device=device),
                sources=sources,plan_sha256=_hash_file(output/'plan.json'),initialization=init,
                initialization_manifest_sha256=init_manifest_sha256,
                loader='Transformers AutoModelForCausalLM + PEFT get_peft_model; pinned SafeTune KL objective is executed directly')
            _write(output/'run-config.json',config)
            report.update(config_sha256=_hash_file(output/'run-config.json'),plan_sha256=config['plan_sha256'],
                initialization_tensor_sha256=init['tensor_sha256'],adapter_coverage=coverage,stage='preflight')
            _write(output/'report.json',report)
            # All model outputs, including the selected suffix CE forward, are checked.
            hook=model.register_forward_hook(finite_logits_hook)
            parameters=[p for p in model.parameters() if p.requires_grad]
            frozen=[(n,p,p.detach().flatten()[:1024].cpu().clone()) for n,p in model.named_parameters() if not p.requires_grad]
            initial=adapter_state(model)
            report['frozen_parameter_check']=dict(tensors=len(frozen),elements=sum(s.numel() for _,_,s in frozen),
                scope='First up to 1024 elements of every frozen tensor; flags and absence of gradients checked throughout. Not a full in-memory weight hash.')

            def check_base():
                require(all(not p.requires_grad and p.grad is None for _,p,_ in frozen),'Frozen base gradient/trainability changed')
                require(all(p.requires_grad for p in parameters),'Adapter trainability changed')

            def task_grads(row):
                with model_mode(model,training=True,checkpointing=True):
                    loss=task_loss(model,row,device=device)
                    gradients=torch.autograd.grad(loss,parameters,allow_unused=False)
                require(all(torch.isfinite(g).all().item() for g in gradients),'Nonfinite task gradients')
                check_base()
                return loss.detach().item(),[g.detach() for g in gradients]

            def align_grads(row,*,initial_equivalence=False,diagnostics=None):
                batch={k:torch.tensor([row[k]],dtype=torch.long,device=device) for k in ('input_ids','attention_mask')}
                result=alignment_gradients(model,batch,parameters,safegrad_kl_alignment_loss,
                    require_initial_equivalence=initial_equivalence,diagnostics=diagnostics)
                check_base();return result

            # Exact reference check for BOTH arms before the first optimizer update.
            diagnostic={}
            alignment_probe=max(bundle['alignment'],key=lambda r:r['sequence_tokens'])
            align_value,align_gradient=align_grads(alignment_probe,initial_equivalence=True,diagnostics=diagnostic)
            longest=max(bundle['train'],key=lambda r:r['sequence_tokens'])
            task_value,task_gradient=task_grads(longest)
            require(tensor_digest(adapter_state(model))==init['tensor_sha256'],'Preflight changed initialization')
            check_base()
            report['preflight']={**diagnostic,'alignment_id':alignment_probe['id'],'task_id':longest['id'],
                                 'task_loss':task_value,'weights_unchanged':True,'memory':_memory(device)}
            del align_gradient,task_gradient
            model.zero_grad(set_to_none=True);_clear_device(device);_reset_seed(device)
            optimizer=torch.optim.AdamW(parameters,lr=2.5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
            by_id={r['id']:r for r in bundle['train']};align_by_id={r['id']:r for r in bundle['alignment']}
            report['stage']='training';_write(output/'report.json',report)
            with (output/'steps.jsonl').open('x',encoding='utf-8') as log:
                for update in updates:
                    tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    task_sum=GradientMean();align_sum=GradientMean();task_losses=[];alignment_losses=[]
                    for task_id,align_id in zip(update['task_ids'],update['alignment_ids']):
                        loss,gradients=task_grads(by_id[task_id]);task_sum.add(gradients);task_losses.append(loss);del gradients
                        if arm=='safegrad':
                            loss,gradients=align_grads(align_by_id[align_id]);align_sum.add(gradients);alignment_losses.append(loss);del gradients
                    count=len(update['task_ids'])
                    gradients,stats=finalize_gradients(task_sum.mean(count),align_sum.mean(count) if arm=='safegrad' else None)
                    require(len(gradients)==len(parameters),'Final gradient coverage mismatch')
                    for parameter,gradient in zip(parameters,gradients):parameter.grad=gradient
                    check_base()
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True).item()
                    require(_finite_number(norm,nonnegative=True),'Nonfinite preclip norm')
                    for group in optimizer.param_groups:group['lr']=update['learning_rate']
                    optimizer.step()
                    finite=all(torch.isfinite(p).all().item() for p in parameters)
                    require(finite,'Nonfinite adapter weights after optimizer step');check_base()
                    record=dict(update=update['update'],epoch=update['epoch'],epoch_update=update['epoch_update'],
                        task_ids=update['task_ids'],alignment_ids=update['alignment_ids'] if arm=='safegrad' else [],count=count,
                        learning_rate=update['learning_rate'],task_loss=math.fsum(task_losses)/count,
                        alignment_loss=math.fsum(alignment_losses)/count if arm=='safegrad' else None,
                        gradient_statistics=stats,preclip_norm=norm,finite_adapter_weights=finite,
                        seconds=time.perf_counter()-tick,memory=_memory(device))
                    # Independent check before committing the step to its append-only ledger.
                    validate_ledger([record],{'updates':[update]},arm=arm,expected_updates=1)
                    log.write(_json(record)+'\n');log.flush();os.fsync(log.fileno())
                    report['optimizer_steps']+=1;report['last_update']=update['update']
                    _write(output/'report.json',report)
                    del gradients,task_sum,align_sum
            records=_rows(output/'steps.jsonl')
            report['counts']=validate_ledger(records,plan,arm=arm,expected_updates=len(updates))
            final=adapter_state(model)
            changed=[k for k,v in final.items() if not torch.equal(v,initial[k])]
            require(bool(changed),'No adapter tensors changed from initialization')
            report['changed_adapter_tensors']=len(changed)
            report['maximum_adapter_absolute_change']=max((v-initial[k]).abs().max().item() for k,v in final.items())
            require(all(torch.equal(sample,p.detach().flatten()[:1024].cpu()) for _,p,sample in frozen),'Frozen base samples changed')
            report['frozen_samples_unchanged']=True;check_base()
            report['stage']='save_reload';_write(output/'report.json',report)
            optimizer.zero_grad(set_to_none=True)
            report['adapter']=_save_adapter(model,output/'adapter')
            probe_row=bundle['train'][0];before=_probe(model,probe_row,device)
            hook.remove();hook=None
            del optimizer,parameters,frozen,initial,parameter,gradient
            del model;model=None
            _clear_device(device)
            model=_new_model(snapshot,device);verify_adapter(model)
            hook=model.register_forward_hook(finite_logits_hook)
            saved=_load_adapter_files(output/'adapter',report['adapter']);_put_state(model,saved)
            reloaded=adapter_state(model)
            require(final.keys()==reloaded.keys() and all(torch.equal(v,reloaded[k]) for k,v in final.items()),'Fresh-base reload tensors differ')
            after=_probe(model,probe_row,device)
            require(before.shape==after.shape and before.dtype==after.dtype,'Reload probe shape/dtype mismatch')
            difference=(before.float()-after.float()).abs().max().item()
            require(math.isfinite(difference),'Nonfinite reload difference')
            save_file({'before':before,'after':after},str(output/'reload-probe.safetensors'))
            report['reload']=dict(every_adapter_tensor_bitwise_equal=True,probe_id=probe_row['id'],
                probe_input_ids=probe_row['input_ids'][:probe_row['prompt_tokens']],max_abs_logit_difference=difference,
                probe_sha256=_hash_file(output/'reload-probe.safetensors'),probe_logits_finite=True,
                comparison='Finite same prompt last-position logits before and after fresh-base reload; difference recorded without a tolerance claim.')
            require(_hash_file(initialization/'manifest.json')==init_manifest_sha256
                    and init['files']=={name:_hash_file(initialization/name) for name in init['files']},'Shared initialization changed during run')
            report['steps_sha256']=_hash_file(output/'steps.jsonl')
            report['status']='complete' if report['counts']['complete'] and max_updates is None else 'feasibility'
            report['stage']='finished'
        except BaseException as exc:
            report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',stage=report['stage'],
                          error_type=type(exc).__name__,error=str(exc))
            (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        finally:
            if hook is not None:hook.remove()
            model=None
            gc.collect()
            report['elapsed_seconds']=time.perf_counter()-start
            _write(output/'report.json',report)
    return report


def load_training_run(directory:Path,prepared:Path,screened:Path)->dict:
    """CPU-only artifact validation; requires a complete 171-update run.

    Source snapshots are checked against frozen hashes, rather than assuming that
    an evaluator's later checkout is byte-identical to the original trainer.
    """
    from safetensors.torch import load_file
    import torch
    directory=Path(directory)
    require(not (directory/'.writer.lock').exists(),'Training writer lock exists')
    report=_read(directory/'report.json');config=_read(directory/'run-config.json');plan=_read(directory/'plan.json')
    require(report.get('schema_version')==1 and report.get('status')=='complete' and report.get('stage')=='finished'
            and report.get('run_kind')=='full' and config.get('max_updates') is None,'Training run is feasibility/incomplete')
    require(config.get('schema_version')==1 and config.get('settings')==SETTINGS and config.get('arm') in ('plain','safegrad')
            and report.get('arm')==config['arm'],'Training configuration/arm mismatch')
    require(_hash_file(directory/'run-config.json')==report.get('config_sha256'),'Run configuration hash mismatch')
    bundle=load_training_bundle(prepared,screened)
    require(config.get('data')==bundle['bindings'],'Training data binding mismatch')
    expected=freeze_training_plan(bundle)
    require(plan==expected and _hash_file(directory/'plan.json')==config.get('plan_sha256')==report.get('plan_sha256'),'Training plan mismatch')
    snapshot,files=_snapshot(MODEL,REVISION,True)
    require(config.get('checkpoint_files')==files,'Pinned checkpoint files changed')
    expected_inventory=expected_adapter_inventory(_read(snapshot/'config.json'))
    sources=config.get('sources',{})
    expected_sources={'train.py','numerical-helpers.py','guard.py','prepare.py','tiny-config.json','safetune-safegrad.py','safetune-model-utils.py'}
    require(set(sources)==expected_sources and set(p.name for p in (directory/'sources').iterdir())==expected_sources,'Source snapshot inventory mismatch')
    require(all(_hash_file(directory/'sources'/name)==digest for name,digest in sources.items()),'Source snapshot hash mismatch')
    from experiments.multilingual_repair.train import PACKAGES,SAFETUNE_REVISION
    environment=config.get('environment',{})
    require(environment.get('packages')==PACKAGES and environment.get('SafeTune')==SAFETUNE_REVISION
            and isinstance(environment.get('python'),str) and environment['python'].startswith('3.12.'),'Unapproved recorded training environment')
    require(environment.get('source_hashes')=={'src/safetune/runner/utils/model_utils.py':sources['safetune-model-utils.py'],
            'src/safetune/core/optim/safegrad.py':sources['safetune-safegrad.py']},'SafeTune source binding mismatch')
    require(config.get('platform',{}).get('device')=='mps','Unapproved recorded training device')
    init=config['initialization'];init_dir=directory/'initialization'
    require(_hash_file(init_dir/'manifest.json')==config.get('initialization_manifest_sha256') and _read(init_dir/'manifest.json')==init,'Initialization manifest mismatch')
    require(init.get('zero_B') is True and init.get('schema_version')==1 and init.get('model')==MODEL and init.get('revision')==REVISION,'Initialization pin mismatch')
    require(set(p.name for p in init_dir.iterdir())=={'manifest.json','adapter_config.json','adapter_model.safetensors'},'Initialization inventory mismatch')
    initial=_load_initialization_files(init_dir,init)
    require(init['tensor_inventory']==expected_inventory,'Initial adapter does not cover pinned checkpoint architecture')
    require(all(torch.count_nonzero(v).item()==0 for k,v in initial.items() if 'lora_B' in k),'Initial B tensors changed')
    require(report.get('initialization_tensor_sha256')==init['tensor_sha256'],'Initial tensor binding mismatch')
    state=_load_adapter_files(directory/'adapter',report['adapter'])
    require(report['adapter']['tensor_inventory']==expected_inventory,'Final adapter does not cover pinned checkpoint architecture')
    require(state.keys()==initial.keys() and all(state[k].shape==initial[k].shape and state[k].dtype==initial[k].dtype for k in state),'Final adapter coverage mismatch')
    changed=[k for k,v in state.items() if not torch.equal(v,initial[k])]
    require(bool(changed) and report.get('changed_adapter_tensors')==len(changed),'Final adapter unchanged or change count mismatch')
    maximum=max((v-initial[k]).abs().max().item() for k,v in state.items())
    require(maximum==report.get('maximum_adapter_absolute_change'),'Adapter change evidence mismatch')
    require(_hash_file(directory/'steps.jsonl')==report.get('steps_sha256'),'Training ledger hash mismatch')
    counts=validate_ledger(_rows(directory/'steps.jsonl'),plan,arm=config['arm'],expected_updates=171)
    require(report.get('optimizer_steps')==171 and report.get('counts')==counts,'Training completeness/count mismatch')
    preflight=report.get('preflight',{})
    require(all(preflight.get(k) is True for k in ('initial_lora_B_zero','student_requires_grad','reference_logits_equal','initial_alignment_gradients_zero','weights_unchanged'))
            and preflight.get('initial_alignment_loss')==0. and preflight.get('max_abs_logit_difference')==0.
            and preflight.get('status')=='passed' and _finite_number(preflight.get('raw_pinned_helper_kl')),'Missing exact initial reference check')
    require(report.get('frozen_samples_unchanged') is True,'Missing frozen base sample check')
    reload=report.get('reload',{})
    require(reload.get('every_adapter_tensor_bitwise_equal') is True and reload.get('probe_logits_finite') is True
            and _hash_file(directory/'reload-probe.safetensors')==reload.get('probe_sha256'),'Missing/tampered reload evidence')
    probes=load_file(str(directory/'reload-probe.safetensors'),device='cpu')
    require(set(probes)=={'before','after'} and probes['before'].shape==probes['after'].shape
            and probes['before'].dtype==probes['after'].dtype and all(torch.isfinite(v).all().item() for v in probes.values()),'Invalid reload logits')
    require((probes['before'].float()-probes['after'].float()).abs().max().item()==reload.get('max_abs_logit_difference'),'Reload difference evidence mismatch')
    probe=bundle['train'][0]
    require(reload.get('probe_id')==probe['id'] and reload.get('probe_input_ids')==probe['input_ids'][:probe['prompt_tokens']],'Reload prompt mismatch')
    return dict(report=report,config=config,plan=plan,adapter=directory/'adapter',counts=counts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','screened','output','initialization'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=('plain','safegrad'),required=True)
    parser.add_argument('--create-initialization',action='store_true')
    parser.add_argument('--max-updates',type=int)
    args=parser.parse_args(argv)
    report=run_training(args.prepared,args.screened,args.output,arm=args.arm,initialization=args.initialization,
                        create_initialization=args.create_initialization,max_updates=args.max_updates)
    print(_json(report))
    return 0 if report['status'] in ('complete','feasibility') else 1


if __name__=='__main__':raise SystemExit(main())
