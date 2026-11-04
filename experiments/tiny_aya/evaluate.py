"""Freeze 700 Tiny Aya user-only prompts and generate auditable local batches.

This stage generates answers only. References stay in metadata for later judging.
A valid incomplete journal is inspectable, but never a completed evaluation.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import importlib.metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import time

from experiments.tiny_aya import runtime, train

MODEL=train.MODEL
REVISION=train.REVISION
SETTINGS=dict(schema_version=1,model=MODEL,revision=REVISION,input_limit=2048,
    device='mps',dtype='bfloat16',attention='eager',eos_token_ids=[3,6],pad_token_id=0,
    max_new_tokens=512,template='official default, user only, add_generation_prompt=True',
    truncation=False,seed=42)
_JSON=train._json
_digest=train._digest
_hash_file=train._hash_file
_write=train._write
require=train.require


def _pairs(pairs):
    require(len(pairs)==len({k for k,_ in pairs}),'Duplicate JSON object key')
    return dict(pairs)


def _invalid_constant(value):
    raise ValueError('Nonfinite JSON constant')


def _parse(text):
    return json.loads(text,object_pairs_hook=_pairs,parse_constant=_invalid_constant)


def _read(path):
    return _parse(Path(path).read_text(encoding='utf-8'))


def _rows(path):
    blob=Path(path).read_bytes()
    require(not blob or blob.endswith(b'\n'),'Incomplete JSONL terminal newline')
    return [_parse(line) for line in blob.decode('utf-8').splitlines()]


def _same(left,right):
    # Unlike Python equality, canonical JSON distinguishes bool/int/float.
    return _JSON(left)==_JSON(right)


def generation_config():
    """Resolve global defaults; never inherit checkpoint sampling or constraints."""
    return runtime.resolved_generation_config(max_new_tokens=512,
        eos_token_id=[3,6],pad_token_id=0,bos_token_id=2)


def _environment():
    return dict(python=platform.python_version(),packages={name:importlib.metadata.version(name)
        for name in ('torch','transformers','tokenizers','huggingface-hub','safetensors','peft')})


def _tokenizer():
    from transformers import AutoTokenizer
    snapshot,files=train._snapshot(MODEL,REVISION,False)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    return tokenizer,files


def _checkpoint():
    return train._snapshot(MODEL,REVISION,True)


def _source_paths():
    return {'evaluate.py':Path(__file__),'runtime.py':Path(runtime.__file__),
        'train.py':Path(train.__file__),'guard.py':Path(__file__).with_name('guard.py'),
        'prepare.py':Path(__file__).with_name('prepare.py'),'tiny-config.json':Path(__file__).with_name('config.json'),
        'attention-helpers.py':Path(inspect.getfile(train.pin_attention_backend))}


def _snapshot_sources(output):
    sources=output/'sources';sources.mkdir()
    for name,path in _source_paths().items():(sources/name).write_bytes(path.read_bytes())
    return {p.name:_hash_file(p) for p in sorted(sources.iterdir())}


def _verify_sources(directory, sources):
    expected={'evaluate.py','runtime.py','train.py','guard.py','prepare.py','tiny-config.json','attention-helpers.py'}
    require(isinstance(sources,dict) and set(sources)==expected and
        set(p.name for p in (directory/'sources').iterdir())==expected,'Source snapshot inventory mismatch')
    require(all(_hash_file(directory/'sources'/n)==h for n,h in sources.items()),'Source snapshot hash mismatch')


def _verify_executing_sources(sources):
    """Only writers require their current implementation to match the archive."""
    current={name:_hash_file(path) for name,path in _source_paths().items()}
    require(_same(current,sources),'Current implementation source inventory/hashes differ from frozen run')


def _file_entry(path):
    return dict(sha256=_hash_file(path),rows=len(_rows(path)) if path.suffix=='.jsonl' else 1)


def _manifest(directory,status,*,bindings):
    files={p.relative_to(directory).as_posix():_file_entry(p) for p in sorted(directory.rglob('*'))
           if p.is_file() and p.name not in ('manifest.json','.writer.lock')}
    manifest=dict(schema_version=1,status=status,bindings=bindings,files=files)
    _write(directory/'manifest.json',manifest)
    return manifest


def _verify_manifest(directory,*,names,status,bindings):
    manifest=_read(directory/'manifest.json')
    require(manifest.get('schema_version')==1 and manifest.get('status')==status and _same(manifest.get('bindings'),bindings)
        and set(manifest.get('files',{}))==set(names),'Manifest inventory/status/binding mismatch')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
    require(actual==set(names),'Unexpected/missing artifact files')
    for name in names:
        require(_same(manifest['files'][name],_file_entry(directory/name)),f'Artifact hash/count mismatch: {name}')
    return manifest


@contextmanager
def _lock(output,*,fresh=False):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=not fresh)
    path=output/'.writer.lock'
    try:descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise RuntimeError('Generation writer lock exists; inspect its owner before explicitly removing a stale lock') from exc
    try:
        with os.fdopen(descriptor,'w') as handle:
            handle.write(_JSON(dict(pid=os.getpid(),created_unix=time.time()))+'\n');handle.flush();os.fsync(handle.fileno())
        yield
    finally:path.unlink()


def _current_inputs(prepared,screened):
    data=train.load_training_bundle(Path(prepared),Path(screened))
    train._validate_partitions(data)
    tokenizer,files=_tokenizer()
    require(data['bindings']['tokenizer_files']==files,'Tiny tokenizer data binding mismatch')
    rows=[]
    for split,source in (('attack',data['attacks']),('task',data['eval_task'])):
        for original in source:
            # All stored training input IDs and labels are deliberately discarded.
            row={k:copy.deepcopy(v) for k,v in original.items() if k not in
                 ('input_ids','labels','attention_mask','prompt_tokens','sequence_tokens','supervised_tokens','response')}
            if split=='task':row['reference']=original['response']
            row['split']=split
            ids=list(tokenizer.apply_chat_template([dict(role='user',content=row['instruction'])],
                tokenize=True,add_generation_prompt=True,return_dict=False,truncation=False))
            runtime.token_ids(ids,len(tokenizer))
            require(len(ids)<=2048,'Evaluation input exceeds 2048 tokens; truncation is forbidden')
            row.update(input_ids=ids,prompt_tokens=len(ids),input_sha256=_digest(ids))
            rows.append(row)
    require(len(rows)==700 and len({r['id'] for r in rows})==700,'Evaluation must contain exactly 700 unique rows')
    config=dict(schema_version=1,settings=copy.deepcopy(SETTINGS),generation=generation_config().to_dict(),
        data=copy.deepcopy(data['bindings']),tokenizer_files=files,
        template_sha256=_digest(tokenizer.chat_template),effective_template_sha256=_digest(tokenizer.get_chat_template()),
        rows_sha256=_digest(rows),environment=_environment(),
        implementation_parents=dict(guard_review_commit='2ce5ffc',training_review_commit='5ba11a5'))
    return rows,config,tokenizer


def freeze_bundle(prepared:Path,screened:Path,output:Path)->dict:
    rows,config,_=_current_inputs(prepared,screened)
    output=Path(output)
    with _lock(output,fresh=True):
        config['sources']=_snapshot_sources(output)
        _write(output/'run-config.json',config)
        with (output/'items.jsonl').open('x',encoding='utf-8') as handle:
            for row in rows:handle.write(_JSON(row)+'\n')
            handle.flush();os.fsync(handle.fileno())
        report=dict(schema_version=1,status='complete',records=700,attack_records=400,task_records=300,
            config_sha256=_hash_file(output/'run-config.json'),items_sha256=_hash_file(output/'items.jsonl'),
            reference_policy='Metadata only; model input is the official user-only template.')
        _write(output/'report.json',report)
        _manifest(output,'complete',bindings=dict(config_sha256=report['config_sha256'],rows_sha256=config['rows_sha256']))
    return report


def load_bundle(bundle:Path,prepared:Path,screened:Path)->dict:
    """CPU validate frozen rows against current verified sources and tokenizer."""
    bundle=Path(bundle)
    require(not (bundle/'.writer.lock').exists(),'Evaluation bundle writer lock exists')
    rows,expected,tokenizer=_current_inputs(prepared,screened)
    config=_read(bundle/'run-config.json')
    require(set(config)==set(expected)|{'sources'} and all(_same(config[k],v) for k,v in expected.items()),'Frozen evaluation configuration/input binding mismatch')
    _verify_sources(bundle,config['sources'])
    actual=_rows(bundle/'items.jsonl')
    require(_same(actual,rows),'Frozen evaluation rows/tokens/order mismatch')
    report=_read(bundle/'report.json')
    require(_same(report,dict(schema_version=1,status='complete',records=700,attack_records=400,task_records=300,
        config_sha256=_hash_file(bundle/'run-config.json'),items_sha256=_hash_file(bundle/'items.jsonl'),
        reference_policy='Metadata only; model input is the official user-only template.')),'Frozen evaluation report mismatch')
    names={'run-config.json','items.jsonl','report.json'}|{'sources/'+n for n in config['sources']}
    manifest=_verify_manifest(bundle,names=names,status='complete',bindings=dict(config_sha256=report['config_sha256'],rows_sha256=config['rows_sha256']))
    return dict(config=config,report=report,manifest=manifest,rows=actual,tokenizer=tokenizer)


def _training_binding(arm,training_run,prepared,screened,data,checkpoint_files):
    require(arm in ('base','plain','safegrad'),'Unknown generation arm')
    if arm=='base':
        require(training_run is None,'Base arm cannot use an adapter')
        return None,None
    require(training_run is not None,'Adapter arm requires a full validated training run')
    loaded=train.load_training_run(Path(training_run),Path(prepared),Path(screened))
    config,report=loaded['config'],loaded['report']
    require(config.get('arm')==report.get('arm')==arm and _same(config.get('settings'),train.SETTINGS) and
        config.get('max_updates') is None and report.get('status')=='complete' and report.get('run_kind')=='full',
        'Adapter arm/configuration/full-run mismatch')
    require(_same(config.get('data'),data) and _same(config.get('checkpoint_files'),checkpoint_files),'Adapter data/checkpoint mismatch')
    require(isinstance(config.get('initialization'),dict) and config['initialization'].get('tensor_sha256') and
        config.get('initialization_manifest_sha256'),'Missing shared initialization identity')
    identity=dict(arm=arm,config_sha256=_hash_file(Path(training_run)/'run-config.json'),
        report_sha256=_hash_file(Path(training_run)/'report.json'),adapter=report['adapter'],
        initialization_tensor_sha256=config['initialization']['tensor_sha256'],
        initialization_manifest_sha256=config['initialization_manifest_sha256'])
    return identity,loaded['adapter']


def _partition(rows,batch_size):
    require(type(batch_size) is int and batch_size in (1,4),'Tiny generation batch size must be 1 or 4')
    return [[r['id'] for r in rows[i:i+batch_size]] for i in range(0,len(rows),batch_size)]


def _run_settings(frozen,bundle,checkpoint_files,arm,training,batch_size):
    return dict(schema_version=1,arm=arm,settings=copy.deepcopy(SETTINGS),generation=copy.deepcopy(frozen['config']['generation']),
        bundle_manifest_sha256=_hash_file(Path(bundle)/'manifest.json'),bundle_config_sha256=_hash_file(Path(bundle)/'run-config.json'),
        input_rows_sha256=frozen['config']['rows_sha256'],data=frozen['config']['data'],
        tokenizer_files=frozen['config']['tokenizer_files'],checkpoint_files=checkpoint_files,
        training=training,batch_size=batch_size,batches=_partition(frozen['rows'],batch_size),environment=_environment())


def _validate_config(config,expected,output):
    require(set(config)==set(expected)|{'sources'} and all(_same(config[k],v) for k,v in expected.items()),'Generation settings/input/model/adapter binding changed')
    _verify_sources(output,config['sources'])


def _ledger(output,config,frozen,*,batches=None,start_index=0):
    from transformers import GenerationConfig
    ledger=_rows(output/'batches.jsonl') if batches is None else batches
    by_id={row['id']:row for row in frozen['rows']};flat=[]
    require(start_index+len(ledger)<=len(config['batches']),'Too many generation batches')
    generation=GenerationConfig.from_dict(config['generation'])
    config_hash=_hash_file(output/'run-config.json')
    for index,batch in enumerate(ledger,start=start_index):
        members=config['batches'][index]
        require(isinstance(batch,dict) and set(batch)=={'batch_index','item_ids','config_sha256','rows'} and
            type(batch.get('batch_index')) is int and batch['batch_index']==index and
            batch.get('item_ids')==members and batch.get('config_sha256')==config_hash and
            type(batch.get('rows')) is list and len(batch['rows'])==len(members),'Reordered/duplicate/invalid batch journal')
        width=max(len(by_id[i]['input_ids']) for i in members)
        durations=[];suffix_lengths=[];eos_lengths=[]
        for position,(item_id,row) in enumerate(zip(members,batch['rows'])):
            source=by_id[item_id]
            expected=dict(record_id=config['arm']+':'+item_id,item_id=item_id,family_id=source['family_id'],
                lang=source['lang'],split=source['split'],arm=config['arm'],arm_identity_sha256=_digest(config['training']),
                input_sha256=source['input_sha256'],batch_index=index,batch_position=position)
            evidence=runtime.validate_evidence(row,frozen['tokenizer'],source['input_ids'],generation_config=generation,padded_width=width)
            require(set(row)==set(expected)|set(evidence) and all(type(row.get(k)) is type(v) and row[k]==v for k,v in expected.items()),'Generation identity/batch metadata mismatch')
            durations.append(row['elapsed_batch_seconds']);suffix_lengths.append(len(row['raw_generated_ids']))
            eos_lengths.append(row['completion_tokens'])
            flat.append(row)
        require(len(set(durations))==1 and len(set(suffix_lengths))==1,'Inconsistent batch duration/suffix width')
        require(suffix_lengths[0]==max(eos_lengths),'Batch padded generation has impossible terminal width')
    return ledger,flat


def _report(config,rows,*,status=None,error_type=None):
    status=status or ('complete' if len(rows)==700 else 'incomplete')
    require(status in ('complete','incomplete','failed'),'Invalid generation status')
    require((status=='failed' and type(error_type) is str and error_type.isidentifier()) or
        (status!='failed' and error_type is None),'Inconsistent failure status/type')
    require((status=='complete')==(len(rows)==700),'Completion requires all 700 rows')
    report=dict(schema_version=1,status=status,arm=config['arm'],records=len(rows),expected_records=700,
        completed_batches=len({r['batch_index'] for r in rows}),total_batches=len(config['batches']),
        eos_records=sum(r['finish_reason']=='eos' for r in rows),capped_records=sum(r['finish_reason']=='length' for r in rows),
        empty_records=sum(r['empty_response'] for r in rows),config_sha256=hashlib.sha256((_JSON(config)+'\n').encode()).hexdigest(),
        scope='Local generation only; safety, task scoring and human review are pending.')
    if error_type is not None:report['error_type']=error_type
    return report


def _save_state(output,config,rows,*,status=None,error_type=None):
    report=_report(config,rows,status=status,error_type=error_type)
    _write(output/'report.json',report)
    _manifest(output,report['status'],bindings=dict(config_sha256=_hash_file(output/'run-config.json'),
        bundle_manifest_sha256=config['bundle_manifest_sha256'],input_rows_sha256=config['input_rows_sha256'],
        checkpoint_files_sha256=_digest(config['checkpoint_files']),training_sha256=_digest(config['training'])))
    return report


def _device():
    return train._device()


def _new_model(snapshot,device,generation,adapter):
    """Fresh independent cached base; no task/judge model sharing or remote code."""
    import torch
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='eager').to(device)
    train.pin_attention_backend(model,'eager')
    require(model.config.model_type=='cohere2' and model.config.num_hidden_layers==36 and
        model.config.hidden_size==2048 and model.config.vocab_size==262144,'Unexpected pinned Tiny architecture')
    require(train.verify_attention_backend(model,'eager')['attention_modules']==36,'Incorrect eager attention coverage')
    if adapter is not None:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter,is_trainable=True,local_files_only=True)
        train.verify_adapter(model)
    for name,p in model.named_parameters():
        require(p.device.type==device and p.grad is None and torch.isfinite(p).all().item(),'Invalid/nonfinite model tensor')
        require(p.dtype==(torch.float32 if '.lora_' in name else torch.bfloat16),'Unexpected model tensor precision')
        p.requires_grad_(False)
    model.eval()
    # Transformers 5 fills even explicit None options from this model baseline.
    # The model belongs exclusively to this run, and its frozen baseline prevents
    # checkpoint penalties/suppressions/stop rules leaking into those None slots.
    model.generation_config=copy.deepcopy(generation)
    if hasattr(model,'get_base_model'):
        model.get_base_model().generation_config=copy.deepcopy(generation)
    effective,kwargs=model._prepare_generation_config(copy.deepcopy(generation))
    require(effective.to_dict()==generation.to_dict() and not kwargs,'Unfrozen effective generation defaults')
    return model


def generate(bundle:Path,output:Path,*,arm:str,prepared:Path,screened:Path,
             training_run:Path|None=None,batch_size:int=1,max_batches:int|None=None)->dict:
    require(max_batches is None or type(max_batches) is int and max_batches>0,'max_batches must be a positive integer')
    frozen=load_bundle(bundle,prepared,screened)
    partitions=_partition(frozen['rows'],batch_size)
    require(max_batches is None or max_batches<len(partitions),'max_batches must be a strict incomplete batch prefix')
    snapshot,files=_checkpoint()
    training,adapter=_training_binding(arm,training_run,prepared,screened,frozen['config']['data'],files)
    expected=_run_settings(frozen,bundle,files,arm,training,batch_size)
    output=Path(output);existed=output.exists()
    require(not any(output.resolve()==Path(p).resolve() or output.resolve().is_relative_to(Path(p).resolve()) or
        Path(p).resolve().is_relative_to(output.resolve()) for p in (bundle,prepared,screened,*([training_run] if training_run else []))),
        'Generation output and input directories must be independent')
    from transformers import GenerationConfig
    with _lock(output):
        if existed:
            require((output/'run-config.json').is_file(),'Existing output lacks frozen run config')
            config=_read(output/'run-config.json');_validate_config(config,expected,output)
            allowed={'run-config.json','batches.jsonl','report.json','manifest.json','.writer.lock'}|{'sources/'+n for n in config['sources']}
            require(all(p.relative_to(output).as_posix() in allowed for p in output.rglob('*') if p.is_file()),'Unexpected generation artifact')
        else:
            config={**expected,'sources':_snapshot_sources(output)}
            _write(output/'run-config.json',config)
            with (output/'batches.jsonl').open('x') as handle:handle.flush();os.fsync(handle.fileno())
        ledger,rows=_ledger(output,config,frozen)
        pending=list(enumerate(config['batches']))[len(ledger):]
        # max_batches is an absolute prefix boundary, not additional batches.
        if max_batches is not None:pending=[(i,b) for i,b in pending if i<max_batches]
        if not pending:return _save_state(output,config,rows)
        # Never append new outputs under a prior implementation's provenance.
        # Keep this writer-only check separate from archived evidence consumers.
        _verify_executing_sources(config['sources'])
        _save_state(output,config,rows)
        model=None
        try:
            device=_device();generation=GenerationConfig.from_dict(config['generation'])
            train._reset_seed(device)
            model=_new_model(snapshot,device,generation,adapter)
            sources={r['id']:r for r in frozen['rows']}
            for index,members in pending:
                batch_rows=runtime.generate_batch(model,frozen['tokenizer'],[sources[i]['input_ids'] for i in members],
                    generation_config=generation,device=device)
                require(len(batch_rows)==len(members),'Generation batch omitted rows')
                for position,(item_id,row) in enumerate(zip(members,batch_rows)):
                    source=sources[item_id]
                    row.update(record_id=arm+':'+item_id,item_id=item_id,family_id=source['family_id'],lang=source['lang'],
                        split=source['split'],arm=arm,arm_identity_sha256=_digest(training),input_sha256=source['input_sha256'],
                        batch_index=index,batch_position=position)
                batch=dict(batch_index=index,item_ids=members,config_sha256=_hash_file(output/'run-config.json'),rows=batch_rows)
                _,validated=_ledger(output,config,frozen,batches=[batch],start_index=index)
                with (output/'batches.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(_JSON(batch)+'\n');handle.flush();os.fsync(handle.fileno())
                rows.extend(validated)
            # Re-open the complete fsynced journal before finalizing the manifest.
            _,rows=_ledger(output,config,frozen)
            return _save_state(output,config,rows)
        except BaseException as exc:
            _save_state(output,config,rows,status='failed',error_type=type(exc).__name__)
            raise
        finally:
            model=None;gc.collect()
            if 'device' in locals() and device=='mps':
                import torch
                torch.mps.empty_cache()


def load_generation(directory:Path,bundle:Path,prepared:Path,screened:Path,training_run:Path|None=None)->dict:
    """CPU-only full validation. Incomplete prefixes have explicit report status.

    Consumers requiring a scientific evaluation must require complete + 700 rows.
    """
    directory=Path(directory)
    require(not (directory/'.writer.lock').exists(),'Generation writer lock exists')
    frozen=load_bundle(bundle,prepared,screened)
    config=_read(directory/'run-config.json')
    _,files=_checkpoint()
    training,_=_training_binding(config.get('arm'),training_run,prepared,screened,frozen['config']['data'],files)
    expected=_run_settings(frozen,bundle,files,config.get('arm'),training,config.get('batch_size'))
    _validate_config(config,expected,directory)
    _,rows=_ledger(directory,config,frozen)
    report=_read(directory/'report.json')
    rebuilt=_report(config,rows,status=report.get('status'),error_type=report.get('error_type'))
    require(_same(report,rebuilt),'Generation report/count/status mismatch')
    names={'run-config.json','batches.jsonl','report.json'}|{'sources/'+n for n in config['sources']}
    bindings=dict(config_sha256=_hash_file(directory/'run-config.json'),bundle_manifest_sha256=config['bundle_manifest_sha256'],
        input_rows_sha256=config['input_rows_sha256'],checkpoint_files_sha256=_digest(config['checkpoint_files']),training_sha256=_digest(training))
    manifest=_verify_manifest(directory,names=names,status=report['status'],bindings=bindings)
    return dict(config=config,report=report,manifest=manifest,rows=rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for command in ('freeze','generate','verify'):
        sub=commands.add_parser(command)
        for name in ('prepared','screened','output'):sub.add_argument('--'+name,type=Path,required=True)
        if command!='freeze':
            sub.add_argument('--bundle',type=Path,required=True)
            sub.add_argument('--training-run',type=Path)
        if command=='generate':
            sub.add_argument('--arm',choices=('base','plain','safegrad'),required=True)
            sub.add_argument('--batch-size',type=int,choices=(1,4),default=1)
            sub.add_argument('--max-batches',type=int)
    args=vars(parser.parse_args(argv));command=args.pop('command')
    if command=='freeze':result=freeze_bundle(**args)
    elif command=='generate':result=generate(**args)
    else:result=load_generation(args.pop('output'),**args)['report']
    print(_JSON(result))
    return 0


if __name__=='__main__':raise SystemExit(main())
