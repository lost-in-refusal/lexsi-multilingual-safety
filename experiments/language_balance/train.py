"""Train two truthful SafeGrad exposure arms from one immutable initialization.

The numerical implementation is imported from the original frozen experiment.
Full results require1000 exact scheduled updates; shorter runs are feasibility.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import random
import resource
import time
import traceback

from experiments.multilingual_repair import train as v1
from experiments.language_balance import study as study_module
from experiments.language_balance.study import load_study
from experiments.multilingual_repair.train import (
    ROOT, MODEL, REVISION, TARGETS, require, sha_bytes, write_json,
    verify_retokenized, verify_environment, pin_attention_backend, verify_attention_backend,
    model_mode, task_loss, alignment_gradients, combine_gradients, aggregate_calibration)


def select_schedule(study, arm, max_steps=None):
    require(arm in ('control','balanced'), 'Unknown exposure arm')
    if max_steps is not None:
        require(type(max_steps) is int and 2 <= max_steps < 1000, 'Feasibility requires at least2 and fewer than1000 steps')
    steps = 1000 if max_steps is None else max_steps
    require(len(study['schedules'][arm]) == len(study['alignment_order']) == 1000, 'Incomplete frozen schedule')
    return study['schedules'][arm][:steps], study['alignment_order'][:steps], 'fixed_schedule' if max_steps is None else 'feasibility'


def source_hashes():
    return {str(Path(module.__file__).resolve().relative_to(ROOT)): sha_bytes(Path(module.__file__).read_bytes())
            for module in (v1, study_module)} | {'experiments/language_balance/train.py':sha_bytes(Path(__file__).read_bytes())}


def checkpoint_hashes(snapshot):
    paths=sorted(Path(snapshot).glob('*.safetensors'))
    require(bool(paths), 'Missing base checkpoint files')
    return {p.name:sha_bytes(p.read_bytes()) for p in paths}


def initialization_config(study, checkpoint_files):
    return {'original_config_sha256':study['spec']['original_config_sha256'],
            'study_manifest_sha256':study['manifest_sha256'], 'model':MODEL,'model_revision':REVISION,
            'base_checkpoint_files':checkpoint_files}


def tensor_metadata(state, *, require_zero_b):
    import torch
    require(bool(state) and all(isinstance(n,str) and ('lora_A.' in n or 'lora_B.' in n) for n in state), 'Only named LoRA tensors may be captured')
    require(all(isinstance(t,torch.Tensor) and t.dtype==torch.float32 and t.numel()>0 and t.ndim==2
                and torch.isfinite(t).all().item() for t in state.values()), 'Invalid/nonfinite adapter shape/dtype/tensors')
    b=[t for n,t in state.items() if '.lora_B.' in n]
    a=[t for n,t in state.items() if '.lora_A.' in n]
    require(bool(a) and len(a)==len(b), 'Missing paired LoRA A/B tensors')
    if require_zero_b:
        require(all(torch.count_nonzero(t).item()==0 for t in b), 'Initialization must have exactly zero LoRA B tensors')
    return {n:{'shape':list(t.shape),'dtype':str(t.dtype),
               'sha256':sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())}
            for n,t in sorted(state.items())}


def validate_adapter_config(config):
    require(config.get('base_model_name_or_path')==MODEL and config.get('revision')==REVISION
            and config.get('r')==4 and config.get('lora_alpha')==8 and config.get('lora_dropout')==0.
            and config.get('bias')=='none' and config.get('use_dora') is False
            and config.get('use_rslora') is False and not config.get('modules_to_save')
            and not config.get('rank_pattern') and not config.get('alpha_pattern')
            and set(config.get('target_modules',[]))==TARGETS,'Actual adapter configuration differs')


def capture_initialization(model, directory, source_config):
    from peft import get_peft_model_state_dict
    directory=Path(directory);require(not directory.exists(),'Initialization cannot be overwritten')
    state={n:t.detach().cpu().clone() for n,t in get_peft_model_state_dict(model, save_embedding_layers=False).items()}
    metadata=tensor_metadata(state,require_zero_b=True)
    validate_adapter_config(model.peft_config['default'].to_dict())
    directory.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(directory,safe_serialization=True,save_embedding_layers=False)
    files={p.name:sha_bytes(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()}
    require({'adapter_config.json','adapter_model.safetensors'} <= set(files),'Missing initialization adapter artifacts')
    manifest={'schema_version':1,'status':'captured_zero_update_initialization','optimizer_steps':0,
              'source_config':source_config,'source_config_sha256':sha_bytes(json.dumps(source_config,sort_keys=True,allow_nan=False).encode()),
              'zero_lora_B':True,'tensors':metadata,'files':files}
    write_json(directory/'manifest.json',manifest)
    return validate_initialization(directory,source_config)


def validate_initialization(directory, source_config):
    from safetensors.torch import load_file
    directory=Path(directory)
    require(directory.is_dir() and not directory.is_symlink() and (directory/'manifest.json').is_file(), 'Missing initialization manifest')
    raw=(directory/'manifest.json').read_bytes();manifest=json.loads(raw)
    require(manifest.get('schema_version')==1 and manifest.get('status')=='captured_zero_update_initialization'
            and manifest.get('optimizer_steps')==0 and manifest.get('zero_lora_B') is True, 'Initialization is not a zero-update capture')
    require(manifest.get('source_config')==source_config
            and manifest.get('source_config_sha256')==sha_bytes(json.dumps(source_config,sort_keys=True,allow_nan=False).encode()),'Initialization shared source config mismatch')
    require({'adapter_config.json','adapter_model.safetensors'} <= set(manifest.get('files',{})), 'Missing initialization files')
    require(set(p.name for p in directory.iterdir())==set(manifest['files'])|{'manifest.json'},'Initialization file coverage mismatch')
    for name,digest in manifest['files'].items():
        require(Path(name).name==name and name not in ('','.','..'),'Unsafe initialization file name')
        path=directory/name
        require(path.is_file() and not path.is_symlink() and sha_bytes(path.read_bytes())==digest,'Initialization file hash mismatch')
    validate_adapter_config(json.loads((directory/'adapter_config.json').read_bytes()))
    state=load_file(directory/'adapter_model.safetensors',device='cpu')
    require(tensor_metadata(state,require_zero_b=True)==manifest.get('tensors'),'Initialization tensor identity mismatch')
    return {**manifest,'directory':str(directory.resolve()),'manifest_sha256':sha_bytes(raw)}


def load_initialization(model, directory, source_config):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    manifest=validate_initialization(directory,source_config)
    state=load_file(Path(directory)/'adapter_model.safetensors',device='cpu')
    current=get_peft_model_state_dict(model, save_embedding_layers=False)
    require(set(current)==set(state) and all(current[n].shape==t.shape and current[n].dtype==t.dtype for n,t in state.items()),'Initialization name/shape/dtype mismatch')
    set_peft_model_state_dict(model,state)
    actual=get_peft_model_state_dict(model, save_embedding_layers=False)
    require(all(torch.equal(t,actual[n].detach().cpu()) for n,t in state.items()),'Initialization exact load mismatch')
    return manifest


def verify_pool_tokenization(study, tokenizer):
    for row in study['pool']:
        messages=[{'role':'system','content':row['system']},{'role':'user','content':row['request']}]
        prompt=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
        full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':row['reference']}],
                                            tokenize=True,add_generation_prompt=False,return_dict=False)
        target=tokenizer(row['reference'],add_special_tokens=False)['input_ids']
        require(len(target)<=192, f'{row["id"]}: raw target exceeds192 tokens')
        if 'reference_tokens' in row:
            require(row['reference_tokens']==len(target), f'{row["id"]}: stale target token count')
        require(prompt==row['input_ids'][:row['prompt_tokens']] and full==row['input_ids'], f'{row["id"]}: exact repaired pool tokenization mismatch')


def require_finite_tree(value):
    if isinstance(value,float):
        require(math.isfinite(value),'Nonfinite training diagnostic')
    elif isinstance(value,dict):
        for item in value.values():require_finite_tree(item)
    elif isinstance(value,list):
        for item in value:require_finite_tree(item)


def validate_gradient_statistics(stats):
    keys={'dot','task_norm','alignment_norm','conflict','zero_alignment','projection_coefficient','rho','cosine'}
    require(isinstance(stats,dict) and set(stats)==keys,'Incomplete SafeGrad gradient statistics')
    require(all(type(stats[k]) in (float,int) and math.isfinite(stats[k])
                for k in ('dot','task_norm','alignment_norm','projection_coefficient','rho')),
            'Nonfinite/malformed gradient statistics')
    task,alignment,dot=stats['task_norm'],stats['alignment_norm'],stats['dot']
    require(task>=0 and alignment>=0 and stats['rho']==1.,'Invalid gradient statistic norms/weight')
    conflict=dot<0 and alignment>0
    require(stats['conflict'] is conflict and stats['zero_alignment'] is (alignment==0), 'Gradient statistic flags differ')
    coefficient=dot/(alignment*alignment) if conflict else 0.
    require(math.isclose(stats['projection_coefficient'],coefficient,rel_tol=1e-12,abs_tol=0.),'Projection statistic differs')
    cosine=dot/(task*alignment) if task>0 and alignment>0 else None
    require((cosine is None and stats['cosine'] is None)
            or (cosine is not None and type(stats['cosine']) in (float,int)
                and math.isfinite(stats['cosine']) and math.isclose(stats['cosine'],cosine,rel_tol=1e-12,abs_tol=0.)),
            'Cosine statistic differs')


def load_training_run(directory, study, bundle=None):
    """Authenticate a complete fixed schedule before any downstream generation.

    The optional evaluation bundle carries holdout metadata only; the study's
    training-only validated bundle owns original training/config identity.
    """
    from safetensors.torch import load_file
    import torch
    directory=Path(directory)
    require((directory/'report.json').is_file() and not (directory/'report.json').is_symlink(),'Missing training report')
    report=json.loads((directory/'report.json').read_bytes());require_finite_tree(report)
    require(report.get('status')=='passed_training' and report.get('stage')=='complete'
            and report.get('run_kind')=='fixed_schedule' and report.get('training_method')=='safegrad',
            'Evaluation requires successful complete fixed_schedule SafeGrad training')
    arm=report.get('arm');require(arm in ('control','balanced'),'Invalid training exposure arm')
    original=study['bundle'];config=original['config']
    if bundle is not None:
        require(bundle.get('manifest_sha256')==original['manifest_sha256'] and bundle.get('config')==config,
                'Evaluation/training prepared identity differs')
    for key,expected in {'model':MODEL,'model_revision':REVISION,'attention_implementation':'eager',
            'study_manifest_sha256':study['manifest_sha256'],'prepared_manifest_sha256':original['manifest_sha256'],
            'original_config_sha256':sha_bytes(original['blobs']['config.json']),
            'config_sha256':study['manifest']['files']['training-config.json'],
            'training_config':study['spec'],'training_order':study['schedules'][arm],
            'alignment_order':study['alignment_order'],'expected_steps':1000,'optimizer_steps':1000,
            'source_hashes':source_hashes()}.items():
        require(type(report.get(key)) is type(expected) and report[key]==expected,f'Training identity mismatch: {key}')
    for key in ('fresh_optimizer_state','saved_adapter_reload_exact','all_base_parameters_frozen','frozen_parameter_samples_unchanged'):
        require(report.get(key) is True,f'Missing successful training check: {key}')
    require(report.get('changed_adapter_tensors',0)>0 and report.get('maximum_adapter_absolute_change',0)>0,'No trained adapter change')
    expected_optimizer={'name':'AdamW','learning_rate':1e-5,'weight_decay':0.,'betas':[.9,.999],
                        'epsilon':1e-8,'foreach':False,'batch_size':1,'gradient_accumulation_steps':1,'max_gradient_norm':1.}
    require(report.get('optimizer')==expected_optimizer,'Optimizer configuration mismatch')
    numerical=report.get('initial_reference_preflight',{})
    require(numerical.get('status')=='passed' and numerical.get('initial_alignment_loss')==0.
            and all(numerical.get(k) is True for k in ('initial_lora_B_zero','reference_logits_equal','student_requires_grad','initial_alignment_gradients_zero')),
            'Missing exact zero-B initial reference equivalence')
    memory=report.get('memory_preflight',{})
    require(memory.get('status')=='passed' and memory.get('optimizer_steps')==0
            and memory.get('adapter_state_unchanged') is True and memory.get('initial_reference_equivalence_verified') is True,
            'Missing zero-update numerical preflight')
    attention=report.get('attention_backend',{})
    require(attention.get('requested')==attention.get('effective')=='eager' and attention.get('attention_modules',0)>0,'Unverified eager attention backend')
    snapshots={'training-config.json':report['config_sha256'],'original-training-config.json':report['original_config_sha256'],
               'prepared-manifest.json':report['prepared_manifest_sha256'],'study-manifest.json':report['study_manifest_sha256'],
               'training-source.py':report['source_hashes']['experiments/language_balance/train.py'],
               'study-source.py':report['source_hashes']['experiments/language_balance/study.py'],
               'numerical-source.py':report['source_hashes']['experiments/multilingual_repair/train.py']}
    for name,digest in snapshots.items():
        path=directory/name
        require(path.is_file() and not path.is_symlink() and sha_bytes(path.read_bytes())==digest,f'Training source/config snapshot hash mismatch: {name}')
    require(report.get('training_script_sha256')==snapshots['training-source.py'],'Training driver source identity mismatch')
    require(report.get('environment')==verify_environment(),'Pinned numerical environment/source hashes differ')
    checkpoint_files=report.get('base_checkpoint_files')
    require(isinstance(checkpoint_files,dict) and bool(checkpoint_files)
            and all(Path(n).name==n and n.endswith('.safetensors') and isinstance(h,str) and len(h)==64 for n,h in checkpoint_files.items()),'Missing base checkpoint hashes')
    init=report.get('initialization',{})
    require(isinstance(init,dict) and isinstance(init.get('directory'),str),'Missing captured initialization')
    actual_init=validate_initialization(init['directory'],initialization_config(study,checkpoint_files))
    require(init==actual_init,'Initialization report identity mismatch')
    adapter=directory/'adapter';files=report.get('adapter_files')
    require(isinstance(files,dict) and {'adapter_config.json','adapter_model.safetensors'}<=set(files),'Missing adapter artifacts')
    require(adapter.is_dir() and not adapter.is_symlink() and set(p.name for p in adapter.iterdir())==set(files),'Adapter artifact coverage mismatch')
    for name,digest in files.items():
        path=adapter/name
        require(Path(name).name==name and path.is_file() and not path.is_symlink()
                and sha_bytes(path.read_bytes())==digest,'Adapter file hash mismatch')
    adapter_config=json.loads((adapter/'adapter_config.json').read_bytes())
    validate_adapter_config(adapter_config)
    state=load_file(adapter/'adapter_model.safetensors',device='cpu')
    metadata=tensor_metadata(state,require_zero_b=False)
    require(metadata==report.get('adapter_tensors'),'Actual adapter tensor identity mismatch')
    require(set(metadata)==set(init['tensors']) and all(metadata[n]['shape']==t['shape'] and metadata[n]['dtype']==t['dtype']
            for n,t in init['tensors'].items()),'Trained/initial adapter name/shape/dtype mismatch')
    initial_state=load_file(Path(init['directory'])/'adapter_model.safetensors',device='cpu')
    changes=[(state[n]-initial_state[n]).abs().max().item() for n in state]
    require(sum(c>0 for c in changes)==report['changed_adapter_tensors'] and max(changes)==report['maximum_adapter_absolute_change'],
            'Actual adapter changes differ from report')
    steps=directory/'steps.jsonl'
    require(steps.is_file() and not steps.is_symlink() and sha_bytes(steps.read_bytes())==report.get('steps_sha256'),'Step log hash mismatch')
    logs=[json.loads(line) for line in steps.read_bytes().splitlines() if line.strip()]
    require(len(logs)==1000,'Incomplete1000-step training log')
    for number,(row,identity,alignment_id) in enumerate(zip(logs,study['schedules'][arm],study['alignment_order']),1):
        require_finite_tree(row);source=study['pool_by_id'][identity]
        require(row.get('step')==number and row.get('sample_id')==identity and row.get('alignment_id')==alignment_id
                and row.get('split')=='train' and row.get('language')==source['language']
                and row.get('supervised_tokens')==source['supervised_tokens'] and row.get('finite_adapter_weights') is True,
                'Actual logged source/alignment order mismatch')
        require(type(row.get('task_loss')) in (float,int) and row['task_loss']>=0
                and type(row.get('alignment_loss')) in (float,int)
                and type(row.get('gradient_norm_before_clip')) in (float,int) and row['gradient_norm_before_clip']>0
                and isinstance(row.get('gradient_statistics'),dict) and row['gradient_statistics'].get('rho')==1.,
                'Missing finite SafeGrad step diagnostics')
        validate_gradient_statistics(row['gradient_statistics'])
    calibration_ids=[r['id'] for r in original['calibration']]
    require(report.get('calibration_ids')==calibration_ids,'Expected all18 unchanged calibration rows')
    for label in ('calibration_before','calibration_after'):
        measurements=report.get(label,{}).get('measurements',[])
        require([r.get('id') for r in measurements]==calibration_ids,'Calibration measurement coverage mismatch')
        require(v1.aggregate_calibration(measurements)==report[label],'Calibration aggregate mismatch')
        require(all(r.get('language')==s['language'] and r.get('supervised_tokens')==s['supervised_tokens']
                    for r,s in zip(measurements,original['calibration'])),'Calibration original source mismatch')
    return adapter,report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('control', 'balanced'), required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--initialization', type=Path, required=True)
    parser.add_argument('--create-initialization', action='store_true')
    parser.add_argument('--max-steps', type=int, help='Distinctly labeled feasibility prefix, smaller than a complete epoch')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; use a new directory')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {'status': 'running', 'stage': 'verify_prepared', 'arm': args.arm,
              'run_kind': 'feasibility' if args.max_steps is not None else 'fixed_schedule',
              'training_method': 'safegrad', 'started_utc': datetime.now(timezone.utc).isoformat(), 'optimizer_steps': 0,
              'model_weights_loaded': False, 'adapter_directory': 'adapter', 'memory_snapshots': [],
              'scope': 'Public-data adapter experiment; optimization diagnostics do not establish safety or task improvement.',
              'alignment_protocol': 'Independent prompt-only KL(reference||student), temperature1, batchmean; summed positions at batch1.',
              'alignment_stationarity': 'Exactly equal finite reference/student logits use analytic zero KL and zero gradients; no tolerance.',
              'alignment_protocol_limit': 'Different from aligned-response training; no claim of paper-identical protocol.',
              'memory_measurement_scope': 'Synchronized MPS stage samples and process RSS high-water; not a continuous MPS peak.'}

    def save_report():
        write_json(args.output / 'report.json', report)

    def stage(name):
        report['stage'] = name
        save_report()
        print(name, flush=True)

    save_report()
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        study = load_study(args.study, args.prepared)
        bundle = study['bundle']
        config, training = bundle['config'], bundle['config']['training']
        order, alignment_ids, run_kind = select_schedule(study, args.arm, args.max_steps)
        report['run_kind'] = run_kind
        report.update(model=config['model'], model_revision=config['model_revision'],
                      attention_implementation=config['attention_implementation'],
                      training_config=study['spec'], config_sha256=study['manifest']['files']['training-config.json'],
                      original_config_sha256=sha_bytes(bundle['blobs']['config.json']),
                      study_manifest_sha256=study['manifest_sha256'],
                      source_hashes=source_hashes(),
                      prepared_manifest_sha256=bundle['manifest_sha256'],
                      prepared_file_hashes=bundle['manifest']['files'],
                      training_script_sha256=sha_bytes(Path(__file__).read_bytes()),
                      training_order=order, expected_steps=len(order), complete_schedule_steps=1000,
                      optimizer={'name': 'AdamW', 'learning_rate': training['learning_rate'],
                                 'weight_decay': 0.0, 'betas': [0.9, 0.999], 'epsilon': 1e-8,
                                 'foreach': False, 'batch_size': 1, 'gradient_accumulation_steps': 1,
                                 'max_gradient_norm': training['max_grad_norm']})
        (args.output / 'original-training-config.json').write_bytes(bundle['blobs']['config.json'])
        (args.output / 'training-config.json').write_bytes((args.study / 'training-config.json').read_bytes())
        (args.output / 'study-manifest.json').write_bytes((args.study / 'manifest.json').read_bytes())
        (args.output / 'study-source.py').write_bytes(Path(study_module.__file__).read_bytes())
        (args.output / 'numerical-source.py').write_bytes(Path(v1.__file__).read_bytes())
        (args.output / 'prepared-manifest.json').write_bytes(bundle['manifest_bytes'])
        (args.output / 'training-source.py').write_bytes(Path(__file__).read_bytes())
        stage('verify_exact_tokenization')
        tokenizer, snapshot, provenance = verify_retokenized(bundle)
        report['tokenizer'] = provenance
        report['exact_retokenization_rows'] = sum(len(bundle[s]) for s in ('train', 'calibration', 'alignment'))
        verify_pool_tokenization(study, tokenizer)
        del tokenizer
        report['base_checkpoint_files'] = checkpoint_hashes(snapshot)
        stage('verify_training_environment')
        report['environment'] = verify_environment()
        import torch
        import safetune
        from peft import get_peft_model_state_dict
        from safetune.runner.utils.model_utils import load_model, lora_wrap
        from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
        require(torch.backends.mps.is_available(), 'Configured MPS backend is unavailable')
        torch.mps.set_per_process_memory_fraction(training['mps_memory_fraction'])
        report['mps_recommended_max_gib'] = torch.mps.recommended_max_memory() / 2**30
        report['mps_memory_fraction'] = training['mps_memory_fraction']

        def reset_seed():
            safetune.configure(device='mps', seed=config['seed'])
            random.seed(config['seed']); torch.manual_seed(config['seed']); torch.mps.manual_seed(config['seed'])

        def memory(label):
            torch.mps.synchronize()
            record = {'stage': label, 'elapsed_seconds': round(time.perf_counter() - start, 3),
                      'mps_tensor_gib': torch.mps.current_allocated_memory() / 2**30,
                      'mps_driver_gib': torch.mps.driver_allocated_memory() / 2**30,
                      'process_max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30}
            report['memory_snapshots'].append(record)
            return record

        reset_seed()
        stage('load_fresh_frozen_base')
        model = load_model(str(snapshot), dtype='bfloat16', device='mps')
        report['model_weights_loaded'] = True
        report['attention_backend'] = pin_attention_backend(model, config['attention_implementation'])
        require(next(model.parameters()).device.type == 'mps' and next(model.parameters()).dtype == torch.bfloat16,
                'Base model device/dtype differs')
        memory('base_loaded')
        model = lora_wrap(model, r=training['lora_r'], lora_alpha=training['lora_alpha'],
                          lora_dropout=training['lora_dropout'], target_modules=training['target_modules'])
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        require(bool(trainable) and all('lora_' in name for name in trainable), 'Unexpected trainable base parameters')
        require(all(p.dtype == torch.float32 for p in trainable.values()), 'Expected FP32 adapter parameters')
        require(set(model.peft_config['default'].target_modules) == TARGETS, 'Actual LoRA target modules differ')
        parameters = list(trainable.values())
        model.peft_config['default'].base_model_name_or_path = MODEL
        model.peft_config['default'].revision = REVISION
        initialization_source = initialization_config(study, report['base_checkpoint_files'])
        stage('capture_or_verify_shared_initialization')
        if args.create_initialization:
            capture_initialization(model, args.initialization, initialization_source)
        report['initialization'] = load_initialization(model, args.initialization, initialization_source)
        frozen = [(name, p, p.detach().flatten()[:1024].cpu().clone())
                  for name, p in model.named_parameters() if not p.requires_grad]
        initial_adapter = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        report.update(trainable_parameters=sum(p.numel() for p in parameters),
                      trainable_dtypes=sorted({str(p.dtype) for p in parameters}),
                      target_modules=sorted(TARGETS), all_base_parameters_frozen=True,
                      frozen_parameter_samples_checked=len(frozen),
                      frozen_parameter_check_scope='First1024 elements of every frozen tensor; requires_grad and grad checks throughout. Not a full-weight hash.')

        def check_base():
            require(all(not p.requires_grad and p.grad is None for _, p, _ in frozen), 'Frozen base received gradients/trainability')
            require(all(p.requires_grad for p in parameters), 'Adapter trainability was lost')

        def task_grads(row, label):
            with model_mode(model, training=True, checkpointing=True):
                loss = task_loss(model, row, device='mps')
                memory(label + '_task_forward')
                gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
            require(all(torch.isfinite(g).all().item() for g in gradients), 'Nonfinite task gradients')
            value = loss.detach().item()
            memory(label + '_task_backward')
            return value, [g.detach() for g in gradients]

        def align_grads(row, label, *, initial=False, diagnostics=None):
            batch = {key: torch.tensor([row[key]], dtype=torch.long, device='mps')
                     for key in ('input_ids', 'attention_mask')}
            return alignment_gradients(model, batch, parameters, safegrad_kl_alignment_loss,
                                       memory=lambda name: memory(label + '_' + name),
                                       require_initial_equivalence=initial, diagnostics=diagnostics)

        def put_gradients(gradients):
            require(len(gradients) == len(parameters), 'Gradient coverage mismatch')
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            check_base()
            norm = torch.nn.utils.clip_grad_norm_(parameters, training['max_grad_norm'], error_if_nonfinite=True).item()
            require(math.isfinite(norm) and norm > 0, 'Missing/zero/nonfinite combined update')
            return norm

        stage('longest_inputs_memory_preflight')
        longest_task = max(study['pool'], key=lambda r: r['sequence_tokens'])
        longest_alignment = max(bundle['alignment'], key=lambda r: r['sequence_tokens'])
        preflight_start = time.perf_counter()
        report['attention_backend'] = verify_attention_backend(model, config['attention_implementation'])
        report['initial_reference_preflight'] = {'status': 'running', 'sample_id': longest_alignment['id'],
                                                'optimizer_steps': report['optimizer_steps']}
        alignment_loss, alignment = align_grads(longest_alignment, 'preflight', initial=True,
                                                diagnostics=report['initial_reference_preflight'])
        task_value, task = task_grads(longest_task, 'preflight')
        gradients, stats = combine_gradients(task, alignment, rho=training['rho'])
        norm = put_gradients(gradients)
        model.zero_grad(set_to_none=True)
        require(all(torch.equal(initial_adapter[name], p.detach().cpu()) for name, p in trainable.items()),
                'Zero-update memory preflight changed adapter parameters')
        report['memory_preflight'] = {'status': 'passed', 'optimizer_steps': 0, 'adapter_state_unchanged': True,
            'task_id': longest_task['id'], 'task_sequence_tokens': longest_task['sequence_tokens'],
            'alignment_id': longest_alignment['id'],
            'alignment_sequence_tokens': longest_alignment['sequence_tokens'],
            'initial_reference_equivalence_verified': True,
            'task_loss': task_value, 'alignment_loss': alignment_loss, 'gradient_norm_before_clip': norm,
            'gradient_statistics': stats, 'seconds': time.perf_counter() - preflight_start}
        del task, alignment, gradients
        gc.collect(); torch.mps.empty_cache(); reset_seed()
        memory('preflight_cleared')

        calibration_rows = bundle['calibration']
        if args.max_steps is not None:
            calibration_rows = [r for language in ('en', 'ko', 'zh')
                                for r in [x for x in bundle['calibration'] if x['language'] == language][:2]]
        report['calibration_ids'] = [r['id'] for r in calibration_rows]
        report['calibration_scope'] = ('Feasibility subset: first2 prepared calibration rows per language; no selection.'
                                       if args.max_steps is not None else 'All18 frozen calibration rows; no selection.')

        def calibrate(label):
            stage(label)
            measurements = []
            model.zero_grad(set_to_none=True)
            with model_mode(model, training=False, checkpointing=False), torch.no_grad():
                for row in calibration_rows:
                    value = task_loss(model, row, device='mps').item()
                    measurements.append({'id': row['id'], 'language': row['language'], 'loss': value,
                                         'supervised_tokens': row['supervised_tokens']})
            check_base(); memory(label)
            return aggregate_calibration(measurements)

        report['calibration_before'] = calibrate('calibration_before')
        reset_seed()
        alignment_by_id = {r['id']: r for r in bundle['alignment']}
        align_order = [alignment_by_id[i] for i in alignment_ids]
        report['alignment_order'] = alignment_ids
        by_id = study['pool_by_id']
        optimizer = torch.optim.AdamW(parameters, lr=training['learning_rate'], weight_decay=0., foreach=False, betas=(0.9, 0.999), eps=1e-8)
        require(len(optimizer.state) == 0, 'Optimizer must start fresh')
        report['fresh_optimizer_state'] = True
        train_start = time.perf_counter()
        with (args.output / 'steps.jsonl').open('x', encoding='utf-8') as log:
            for number, identity in enumerate(order, 1):
                stage(f'training_step_{number}')
                row = by_id[identity]
                require(row['split'] == 'train', 'Nontraining row entered optimizer')
                optimizer.zero_grad(set_to_none=True)
                step_start = time.perf_counter()
                initial_diagnostic = {} if number == 1 else None
                safety_value, alignment = align_grads(align_order[number - 1], f'step{number}',
                                                       initial=number == 1, diagnostics=initial_diagnostic)
                value, task = task_grads(row, f'step{number}')
                gradients, stats = combine_gradients(task, alignment, rho=training['rho'])
                norm = put_gradients(gradients)
                optimizer.step()
                report['optimizer_steps'] += 1
                finite = all(torch.isfinite(p).all().item() for p in parameters)
                record = {'step': number, 'sample_id': identity, 'split': 'train', 'language': row['language'],
                          'task_loss': value, 'supervised_tokens': row['supervised_tokens'],
                          'alignment_id': report['alignment_order'][number - 1],
                          'alignment_loss': safety_value, 'gradient_statistics': stats,
                          'initial_reference_diagnostic': initial_diagnostic,
                          'gradient_norm_before_clip': norm, 'finite_adapter_weights': finite,
                          'seconds': time.perf_counter() - step_start}
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush(); os.fsync(log.fileno())
                require(finite, 'Nonfinite adapter weights after update')
                check_base(); memory(f'step{number}_optimizer')
                report['last_completed_step'] = record
                save_report()
                del task, alignment, gradients
        report['training_seconds'] = time.perf_counter() - train_start
        require(report['optimizer_steps'] == report['expected_steps'], 'Incomplete fixed training order')
        logged = [json.loads(line) for line in (args.output / 'steps.jsonl').read_text().splitlines()]
        require([row['sample_id'] for row in logged] == order, 'Actual optimizer order differs from frozen order')
        changes = [(p.detach().cpu() - initial_adapter[name]).abs().max().item() for name, p in trainable.items()]
        report.update(changed_adapter_tensors=sum(x > 0 for x in changes), maximum_adapter_absolute_change=max(changes))
        require(report['changed_adapter_tensors'] > 0, 'No adapter tensor changed')
        report['frozen_parameter_samples_unchanged'] = all(torch.equal(saved, p.detach().flatten()[:1024].cpu())
                                                          for _, p, saved in frozen)
        require(report['frozen_parameter_samples_unchanged'], 'Frozen base sample changed')
        check_base(); optimizer.zero_grad(set_to_none=True)
        del optimizer, initial_adapter
        gc.collect(); torch.mps.empty_cache()
        report['calibration_after'] = calibrate('calibration_after')
        stage('save_and_reload_adapter')
        adapter = args.output / 'adapter'
        model.peft_config['default'].base_model_name_or_path = MODEL
        model.peft_config['default'].revision = REVISION
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        original = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()}
        require(bool(original) and all('lora_' in key and torch.isfinite(value).all().item() for key, value in original.items()),
                'Saved state must contain finite adapter tensors only')
        model.load_adapter(adapter, adapter_name='reloaded', is_trainable=False)
        loaded = get_peft_model_state_dict(model, adapter_name='reloaded', save_embedding_layers=False)
        report['saved_adapter_reload_exact'] = (original.keys() == loaded.keys()
            and all(torch.equal(value, loaded[key].detach().cpu()) for key, value in original.items()))
        require(report['saved_adapter_reload_exact'], 'Reloaded adapter tensors differ')
        report['adapter_files'] = {str(path.relative_to(adapter)): sha_bytes(path.read_bytes())
                                  for path in sorted(adapter.rglob('*')) if path.is_file()}
        require({'adapter_config.json', 'adapter_model.safetensors'} <= report['adapter_files'].keys(), 'Missing adapter artifacts')
        report['adapter_weights_bytes'] = (adapter / 'adapter_model.safetensors').stat().st_size
        report['adapter_tensors'] = tensor_metadata(original, require_zero_b=False)
        report['steps_sha256'] = sha_bytes((args.output / 'steps.jsonl').read_bytes())
        report['maximum_observed_stage_mps_driver_gib'] = max(r['mps_driver_gib'] for r in report['memory_snapshots'])
        report.update(status='passed_feasibility' if args.max_steps is not None else 'passed_training', stage='complete')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error_type=type(exc).__name__, error='Details retained in local error.log')
        (args.output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.perf_counter() - start
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_report()
        print(report['status'], flush=True)
    return 0 if report['status'] in ('passed_training', 'passed_feasibility') else 1


if __name__ == '__main__':
    raise SystemExit(main())
