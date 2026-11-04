"""Bounded original-response Qwen triage, never data or training approval.

Readers reconstruct raw evidence on CPU. Only score() may load a model, and
only from the pinned local cache. Independent source audit and human QC remain
pending regardless of the number of triage-selected families.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import gc
import os
from pathlib import Path
import re
import time

from experiments.tiny_aya import guard, prepare, quality, runtime, train

require = quality.require
_read, _rows, _same = quality._read, quality._rows, quality._same
_json, _digest, _hash_file, _write = quality._json, quality._digest, quality._hash_file, quality._write
PROFILE = 'qwen3-14b'
TARGET, PILOT, INITIAL_POOL, EXTENDED_POOL = 900, 200, 2000, 5000
INITIAL_CANDIDATES_SHA256 = '4ac3afbab5a93444a386d1ccbe0e6f9b2cb1b956d2e77ec601de8222c3080987'
# A relocated self-contained evidence copy supplies its own enclosing root.
EVIDENCE_ROOT = Path(__file__).resolve().parents[2]/'results'
REVIEW_FILES = {'report.json','ko.jsonl','zh.jsonl','annotations-ko.json','annotations-zh.json','report-ko.json','report-zh.json'}
AUXILIARY_FILES = ('config.json','attacks.jsonl','alignment.jsonl','translation_review.jsonl')
SOURCE_NAMES = quality.SOURCE_NAMES | {'curate.py'}


def _evidence_root():
    return Path(os.environ.get('TINY_AYA_EVIDENCE_ROOT', EVIDENCE_ROOT)).resolve()


def _resolve(base, relative):
    require(type(relative) is str and bool(relative) and not Path(relative).is_absolute(), 'Expected relative evidence path')
    path = (Path(base)/relative).resolve()
    require(path.is_relative_to(_evidence_root()) and path.is_file(), 'Evidence path escapes configured root or is missing')
    return path


def _reference(base, value):
    require(type(value) is dict and set(value) == {'path','sha256'} and
            type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}',value['sha256']), 'Invalid evidence reference')
    path = _resolve(base,value['path'])
    require(_hash_file(path) == value['sha256'], 'Evidence hash mismatch')
    return path


def _crosslinks(mapping, paths):
    """Historical map keys are labels; compare unambiguous filenames and hashes."""
    require(type(mapping) is dict and len(mapping) == len(paths), 'Evidence cross-link inventory mismatch')
    expected = {p.name:_hash_file(p) for p in paths}
    require(len(expected) == len(paths) and all(type(k) is str and not Path(k).is_absolute() for k in mapping) and
            len({Path(k).name for k in mapping}) == len(mapping) and
            _same({Path(k).name:v for k,v in mapping.items()},expected), 'Evidence cross-link hash mismatch')


def load_readiness(path: Path) -> dict:
    """Validate the scoped, explicitly unqualified route and raw calibration."""
    path = Path(path).resolve()
    require(path.is_relative_to(_evidence_root()), 'Readiness is outside configured evidence root')
    record = _read(path)
    fixed = dict(schema_version=1,status='provisional_triage_ready',allowed_use='candidate_triage_only',
        calibration_qualification='not_established',human_review='pending',judge_profile=PROFILE,
        rubric_version=2,model=quality.QWEN_MODEL,revision=quality.QWEN_REVISION)
    require(type(record) is dict and set(fixed)|{'evidence','root_decision','data_acceptance'} <= set(record) and
        set(record) <= set(fixed)|{'evidence','root_decision','data_acceptance','created_at_utc','limitations'} and
        all(_same(record.get(k),v) for k,v in fixed.items()), 'Invalid triage-only readiness ruling/profile')
    if 'created_at_utc' in record:
        stamp = record['created_at_utc']
        require(type(stamp) is str, 'Invalid readiness timestamp')
        parsed = datetime.fromisoformat(stamp.replace('Z','+00:00'))
        require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, 'Readiness timestamp must be UTC')
    if 'limitations' in record:
        require(type(record['limitations']) is list and all(type(s) is str and bool(s.strip()) for s in record['limitations']), 'Invalid limitations')
    decision = record['root_decision']
    require(type(decision) is dict and set(decision) == {'route','qualification_ruling','decision_basis'} and
        decision['route'] == 'unchanged_qwen_triage_then_independent_source_audit' and
        decision['qualification_ruling'] == 'not_established' and type(decision['decision_basis']) is str and
        bool(decision['decision_basis'].strip()), 'Invalid root triage decision')
    require(_same(record['data_acceptance'],dict(requires_independent_blinded_train_audit=True,
        families=30,languages=['en','ko','zh'],fail_if_incorrect_per_language_gt=3,max_repairs=1)), 'Changed independent-audit policy')
    evidence = record['evidence']
    require(type(evidence) is dict and set(evidence) == {'calibration','prepared','review_evidence','frozen_policy',
        'source_semantic_audit','qwen_semantic_assessment','root_assessment'}, 'Invalid readiness evidence inventory')
    paths = {k:_reference(path.parent,v) for k,v in evidence.items() if k != 'review_evidence'}
    review = evidence['review_evidence']
    require(type(review) is dict and set(review) == REVIEW_FILES, 'Invalid review evidence inventory')
    review_paths = {k:_reference(path.parent,v) for k,v in review.items()}
    require(all(p.name == k for k,p in review_paths.items()) and len({p.parent for p in review_paths.values()}) == 1 and
        paths['calibration'].name == paths['prepared'].name == 'manifest.json', 'Evidence filenames/directories mismatch')
    policy, audit, semantic, root = (_read(paths[k]) for k in ('frozen_policy','source_semantic_audit','qwen_semantic_assessment','root_assessment'))
    identity = dict(model=quality.QWEN_MODEL,revision=quality.QWEN_REVISION,rubric_version=2)
    require(all(_same(policy.get(k),v) and _same(root.get(k),v) for k,v in identity.items()) and
        policy.get('status') == 'frozen_before_qwen_outputs' and policy.get('human_validation') == 'pending' and
        policy.get('semantic_audit_sha256') == evidence['source_semantic_audit']['sha256'] and
        policy.get('original_reference_labels_unchanged') is True and policy.get('report_all_original_confusion_counts') is True,
        'Frozen policy/model/source-audit binding mismatch')
    require(type(policy.get('confirmed_material_error_rows')) is list and len(policy['confirmed_material_error_rows']) == 7 and
        len({r['row_id'] for r in policy['confirmed_material_error_rows']}) == 7 and
        _same(policy.get('criteria'),dict(all_30_completed=True,zero_confirmed_error_rows_labeled_usable=True,
            prior_usable_retention_each_language_at_least=.5,at_least_one_family_accepted=True,
            root_inspects_matching_rejection_evidence=True)), 'Frozen seven-error policy changed')
    require(audit.get('status') == 'provisional_automated_semantic_audit_not_human_validation' and audit.get('human_review') == 'pending' and
        semantic.get('status') == 'provisional_automated_semantic_assessment_not_human_validation', 'Invalid provisional semantic evidence')
    calibration_dir = paths['calibration'].parent
    _crosslinks(semantic.get('source_sha256'),[paths['frozen_policy'],paths['source_semantic_audit']]+
        [calibration_dir/n for n in ('inputs.jsonl','families.jsonl','run-config.json','report.json')])
    _crosslinks(semantic.get('source_annotation_sha256'),[review_paths[n] for n in ('annotations-ko.json','annotations-zh.json')])
    require(root.get('status') == 'not_qualified_for_autonomous_quality_approval' and
        root.get('full_training_approved') is False and root.get('human_validation') == 'pending', 'Root qualification remains unestablished')
    _crosslinks(root.get('evidence'),[paths['frozen_policy'],paths['qwen_semantic_assessment'],paths['calibration']])
    calibration = quality.load_calibration(calibration_dir,paths['prepared'].parent,next(iter(review_paths.values())).parent)
    settings = calibration['config']['settings']
    require(settings.get('judge_profile') == PROFILE and settings.get('rubric_version') == 2 and
        calibration['report']['status'] == 'complete' and len(calibration['rows']) == 30 and
        len(calibration['families']) == 30, 'Readiness requires completed raw-valid Qwen/v2 calibration')
    exclusions = [f['family_id'] for f in calibration['families']]
    require(len(set(exclusions)) == 30, 'Calibration exclusions must be thirty unique families')
    return dict(record=record,sha256=_hash_file(path),paths=paths,calibration=calibration,
        exclusions=exclusions,exclusions_sha256=_digest(exclusions))


def _tiny_tokenizer():
    from transformers import AutoTokenizer
    snapshot, files = train._snapshot(train.MODEL,train.REVISION,False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    require(type(tokenizer.get_chat_template()) is str and bool(tokenizer.get_chat_template()), 'Missing pinned Tiny template')
    return tokenizer,files


def _source_paths():
    return {**quality._source_paths(),'curate.py':Path(__file__)}


def _snapshot_sources(output):
    (output/'sources').mkdir()
    for name,path in _source_paths().items():(output/'sources'/name).write_bytes(path.read_bytes())
    return {n:_hash_file(output/'sources'/n) for n in sorted(SOURCE_NAMES)}


def _verify_sources(output,sources,*,executing=False):
    require(type(sources) is dict and set(sources) == SOURCE_NAMES and
        {p.name for p in (output/'sources').iterdir()} == SOURCE_NAMES and
        all((output/'sources'/n).is_file() and not (output/'sources'/n).is_symlink() and _hash_file(output/'sources'/n) == h for n,h in sources.items()), 'Archived triage source snapshot mismatch')
    if executing:
        require(all(_hash_file(_source_paths()[n]) == h for n,h in sources.items()), 'Current executing triage source differs from frozen source')


def _validate_config(output,config,expected,*,executing=False):
    require(set(config)==set(expected)|{'sources'} and all(_same(config[k],v) for k,v in expected.items()), 'Frozen triage config/source/input changed')
    _verify_sources(output,config['sources'],executing=executing)
    if config['extension_of'] is not None:
        parent=_resolve(output,config['extension_of']['path']).parent
        require(_same(config['sources'],_read(parent/'run-config.json')['sources']), 'Extension executing-source lineage changed')


def _pool(prepared,readiness,*,initial=None):
    manifest,families = guard._prepared(prepared)
    require(len(families) == (INITIAL_POOL if initial is None else EXTENDED_POOL), 'Wrong bounded source pool size')
    if initial is None:
        require(_hash_file(prepared/'candidates.jsonl') == INITIAL_CANDIDATES_SHA256, 'Initial source pool hash mismatch')
        anchor = readiness['paths']['prepared'].parent
        original = _rows(anchor/'candidates.jsonl')
        require(len(original) == 3000 and _same([r for f in families[:1000] for r in f],original), 'Original 1,000-family source prefix changed')
    else:
        anchor = initial
        original = _rows(initial/'candidates.jsonl')
        require(_same([r for f in families[:INITIAL_POOL] for r in f],original), 'Extension source prefix changed')
    old = prepare.verify_bundle(anchor)
    require(all((prepared/n).read_bytes() == (anchor/n).read_bytes() for n in AUXILIARY_FILES) and
        all(_same(manifest.get(k),old.get(k)) for k in ('source_files','curation')), 'Pool auxiliary/config/source provenance changed')
    exclusions = set(readiness['exclusions'])
    require(exclusions <= {f[0]['family_id'] for f in families[:INITIAL_POOL]}, 'Calibration exclusions missing from original pool')
    return manifest,[dict(family_id=f[0]['family_id'],rows=f,source_index=i) for i,f in enumerate(families)
        if f[0]['family_id'] not in exclusions and (initial is None or i >= INITIAL_POOL)]


def _preflight(rows,tokenizer):
    tokenized=[]; reasons=[]
    for row in rows:
        try:
            value = guard.tokenize_row(tokenizer,row)
            runtime.token_ids(value['input_ids'],len(tokenizer))
            tokenized.append(value); reasons.append(None)
        except ValueError as exc:
            tokenized.append(None); reasons.append(str(exc))
    return dict(rows=tokenized,reasons=reasons,passed=all(v is None for v in reasons))


def _inputs(families,tokenizer,tiny):
    frozen = quality._freeze_inputs({'families':families},tokenizer,rubric_version=2,judge_profile=PROFILE)
    for item,source in zip(frozen,families):
        item['source_index']=source['source_index']; item['tiny_preflight']=_preflight(item['rows'],tiny)
        item['input_sha256']=_digest({k:v for k,v in item.items() if k != 'input_sha256'})
    return frozen


def _config(prepared,readiness_path,readiness,output,families,tokenizer,tokenizer_files,checkpoint_files,tiny,tiny_files,extension):
    bindings = dict(prepared_manifest_sha256=_hash_file(prepared/'manifest.json'),
        prepared_candidates_sha256=_hash_file(prepared/'candidates.jsonl'),
        readiness=dict(path=os.path.relpath(readiness_path.resolve(),output.resolve()),sha256=readiness['sha256']),
        frozen_policy_sha256=readiness['record']['evidence']['frozen_policy']['sha256'],
        source_semantic_audit_sha256=readiness['record']['evidence']['source_semantic_audit']['sha256'],
        qwen_semantic_assessment_sha256=readiness['record']['evidence']['qwen_semantic_assessment']['sha256'],
        exclusions=readiness['exclusions'],exclusions_sha256=readiness['exclusions_sha256'])
    config = quality._settings(dict(bindings=bindings),tokenizer,tokenizer_files,checkpoint_files,families,rubric_version=2,judge_profile=PROFILE)
    config['settings']['purpose']='bounded_candidate_triage_only'
    config.update(target=TARGET,pilot=PILOT,source_ceiling=EXTENDED_POOL if extension else INITIAL_POOL,
        tiny=dict(model=train.MODEL,revision=train.REVISION,tokenizer_files=tiny_files,template=tiny.get_chat_template(),
            template_inventory_sha256=_digest(tiny.chat_template),prompt_limit=1024,suffix_limit=512,sequence_limit=2048),extension_of=extension)
    return config


@contextmanager
def _lock(output):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'.writer.lock'
    try: descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc: raise RuntimeError('Triage writer lock exists; inspect owner before explicitly removing a stale lock') from exc
    try:
        with os.fdopen(descriptor,'w') as handle:
            handle.write(_json(dict(pid=os.getpid(),created_unix=time.time()))+'\n');handle.flush();os.fsync(handle.fileno())
        yield
    finally:path.unlink()


def _ledger(output,config,families,tokenizer,*,records=None,start_index=0,inherited=()):
    from transformers import GenerationConfig
    generation=GenerationConfig.from_dict(config['generation'])
    rows=_rows(output/'families.jsonl') if records is None else records
    require(start_index+len(rows) <= len(families), 'Too many triage journal families')
    config_hash=_hash_file(output/'run-config.json')
    passes=sum(r['triage_selected'] for r in inherited)
    for index,row in enumerate(rows,start_index):
        source=families[index]
        require(passes < TARGET, 'Journal contains records after terminal target')
        require(type(row) is dict and set(row) == {'family_index','source_index','family_id','config_sha256','input_sha256',
            'state','evidence','judgment','triage_selected','rejection_reason'}, 'Invalid whole-family triage record')
        require(type(row['family_index']) is int and row['family_index'] == len(inherited)+index and
            type(row['source_index']) is int and row['source_index'] == source['source_index'] and
            row['family_id'] == source['family_id'] and row['input_sha256'] == source['input_sha256'] and
            row['config_sha256'] == config_hash, 'Journal source/config/order mismatch')
        if not source['tiny_preflight']['passed']:
            state='tiny_preflight_reject'; judgment=quality._unknown(state)
        elif len(source['input_ids']) > 4096:
            state='quality_input_cap'; judgment=quality._unknown(state)
        else:
            state='generated'
            require(type(row['evidence']) is dict, 'Missing raw generation evidence')
            rebuilt=runtime.validate_evidence(row['evidence'],tokenizer,source['input_ids'],generation_config=generation,padded_width=len(source['input_ids']))
            require(set(row['evidence']) == set(rebuilt), 'Unknown raw runtime evidence keys')
            judgment=quality.parse_judgment(rebuilt['response'],finish_reason=rebuilt['finish_reason'])
        require(row['state'] == state and (state == 'generated' or row['evidence'] is None) and
            _same(row['judgment'],judgment) and _same(row['triage_selected'],judgment['family_pass']) and
            _same(row['rejection_reason'],_rejection(state,judgment)), 'Triage decision disagrees with reconstructed raw evidence')
        passes += judgment['family_pass']
    return rows


def _rejection(state,judgment):
    if judgment['family_pass']:return None
    if state != 'generated':return state
    if judgment['state'] == 'unknown':return judgment['unknown_reason']
    return 'quality_or_task_equivalence_not_all_pass'


def _report(config,rows,*,failed=None):
    selected=sum(r['triage_selected'] for r in rows)
    total=config['source_ceiling']-len(config['bindings']['exclusions'])
    status=('target_reached' if selected == TARGET else 'pool_exhausted_insufficient' if len(rows) == total else
        'pilot_complete' if len(rows) == PILOT else 'incomplete')
    return dict(schema_version=1,status='failed' if failed else status,error_type=failed,
        allowed_use='candidate_triage_only',calibration_qualification='not_established',human_review='pending',full_training_approved=False,
        target=TARGET,source_pool_families=config['source_ceiling'],excluded_families=len(config['bindings']['exclusions']),
        families_inspected=len(rows),families_pending=total-len(rows),triage_selected=selected,
        generated_families=sum(r['state']=='generated' for r in rows),
        generation_batch_seconds=sum((r['evidence'] or {}).get('elapsed_batch_seconds',0) for r in rows),
        triage_selection_rate=selected/len(rows) if rows else None,
        tiny_preflight_rejects=sum(r['state']=='tiny_preflight_reject' for r in rows),
        quality_input_caps=sum(r['state']=='quality_input_cap' for r in rows),
        output_caps=sum((r['evidence'] or {}).get('finish_reason')=='length' for r in rows),
        unknown_families=sum(r['judgment']['state']=='unknown' for r in rows),
        labels={lang:dict(Counter(r['judgment']['labels'][lang] for r in rows)) for lang in quality.LANGUAGES},
        last_source_index=rows[-1]['source_index'] if rows else None,
        selection_bias='False rejections and task-equivalence errors may distort the selected task distribution.')


def _artifact_names(config,pilot):
    return {'run-config.json','inputs.jsonl','families.jsonl','report.json'} | {'sources/'+n for n in config['sources']} | ({'pilot-report.json'} if pilot else set())


def _pilot(output,config,rows):
    if config['extension_of'] is not None:
        parent=_resolve(output,config['extension_of']['path']).parent
        config=_read(parent/'run-config.json')
    report=_report(config,rows[:PILOT])
    report['prefix_sha256']=_digest(rows[:PILOT])
    return report


def _save(output,config,rows,*,failed=None):
    report=_report(config,rows,failed=failed)
    if len(rows) >= PILOT:
        pilot=_pilot(output,config,rows)
        if (output/'pilot-report.json').exists():require(_same(_read(output/'pilot-report.json'),pilot),'Frozen pilot prefix changed')
        else:_write(output/'pilot-report.json',pilot)
    _write(output/'report.json',report)
    names=_artifact_names(config,len(rows)>=PILOT)
    _write(output/'manifest.json',dict(schema_version=1,status=report['status'],config_sha256=_hash_file(output/'run-config.json'),
        files={n:quality._file_entry(output/n) for n in sorted(names)}))
    return report


def _verify_manifest(output,config,rows,*,writing=False):
    report=_read(output/'report.json')
    failed=report.get('error_type') if report.get('status')=='failed' else None
    require(failed is None or type(failed) is str and bool(failed), 'Invalid failure metadata')
    require(_same(report,_report(config,rows,failed=failed)), 'Triage report reconstruction mismatch')
    if len(rows)>=PILOT:require(_same(_read(output/'pilot-report.json'),_pilot(output,config,rows)), 'Frozen pilot prefix mismatch')
    manifest=_read(output/'manifest.json'); names=_artifact_names(config,len(rows)>=PILOT)
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for p in output.rglob('*')), 'Triage artifacts must be frozen files, not symlinks')
    require(actual == names|{'manifest.json'}|({'.writer.lock'} if writing else set()) and
        _same(manifest,dict(schema_version=1,status=report['status'],config_sha256=_hash_file(output/'run-config.json'),
            files={n:quality._file_entry(output/n) for n in sorted(names)})), 'Triage manifest inventory/hash/count mismatch')
    return report,manifest


def _context(prepared,readiness_path,output,*,extension_of=None):
    readiness=load_readiness(readiness_path)
    inherited=[]; inherited_families=[]; extension=None; initial=None
    if extension_of is not None:
        extension_of=Path(extension_of).resolve()
        require(not (output.resolve().is_relative_to(extension_of) or extension_of.is_relative_to(output.resolve())), 'Extension cycle or overlapping output')
        old_config=_read(extension_of/'run-config.json')
        require(old_config.get('extension_of') is None, 'Only one source extension is permitted')
        # The original prepared directory is supplied by a confined relative reference.
        initial=_resolve(extension_of,old_config['prepared_path']).parent
        previous=load_scores(extension_of,initial)
        require(previous['report']['status']=='pool_exhausted_insufficient' and previous['config']['source_ceiling']==INITIAL_POOL,
            'Extension requires closed exhausted initial pool')
        require(previous['config']['bindings']['readiness']['sha256']==readiness['sha256'], 'Extension readiness changed')
        inherited=previous['rows']; inherited_families=previous['families']
        extension=dict(path=os.path.relpath(extension_of/'manifest.json',output.resolve()),sha256=_hash_file(extension_of/'manifest.json'),
            prepared=dict(path=os.path.relpath(initial/'manifest.json',output.resolve()),sha256=_hash_file(initial/'manifest.json')),
            inherited_rows_sha256=_digest(inherited),config_sha256=_hash_file(extension_of/'run-config.json'))
    _,families=_pool(prepared,readiness,initial=initial)
    tokenizer,tokenizer_files=quality._tokenizer(judge_profile=PROFILE)
    snapshot,checkpoint_files=quality._checkpoint(judge_profile=PROFILE)
    tiny,tiny_files=_tiny_tokenizer()
    families=_inputs(families,tokenizer,tiny)
    config=_config(prepared,readiness_path,readiness,output,families,tokenizer,tokenizer_files,checkpoint_files,tiny,tiny_files,extension)
    config['prepared_path']=os.path.relpath((prepared/'manifest.json').resolve(),output.resolve())
    if extension:
        for key in ('settings','rubric','rubric_sha256','judgment_schema','tokenizer_files','template','template_sha256',
                    'template_inventory_sha256','checkpoint_files','generation','environment','tiny'):
            require(_same(config[key],previous['config'][key]), 'Extension model/rubric/runtime binding changed')
    return config,families,tokenizer,snapshot,inherited,inherited_families


def score(prepared: Path,triage_readiness: Path,output: Path,max_families: int|None=None,*,extension_of: Path|None=None) -> dict:
    """Run/resume a cumulative inspected prefix; initial operational pilot is 200."""
    require(max_families is None or type(max_families) is int and max_families > 0, 'Invalid cumulative family budget')
    prepared,triage_readiness,output=Path(prepared),Path(triage_readiness),Path(output)
    require(all(not (output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()))
        for p in (prepared,triage_readiness.parent)), 'Triage output and inputs must be independent')
    expected,families,tokenizer,snapshot,inherited,_=_context(prepared,triage_readiness,output,extension_of=extension_of)
    existed=output.exists()
    with _lock(output):
        if existed:
            config=_read(output/'run-config.json')
            _validate_config(output,config,expected,executing=True)
            require(_same(_rows(output/'inputs.jsonl'),families),'Frozen triage input/prompt/token mismatch')
            # A validated manifest is required before resuming; do not repair journals.
            old_rows=_ledger(output,config,families,tokenizer,inherited=inherited)
            _verify_manifest(output,config,inherited+old_rows,writing=True)
        else:
            config={**expected,'sources':_snapshot_sources(output)}
            _validate_config(output,config,expected,executing=True)
            _write(output/'run-config.json',config)
            for name,values in (('inputs.jsonl',families),('families.jsonl',[])):
                with (output/name).open('x',encoding='utf-8') as handle:
                    for value in values:handle.write(_json(value)+'\n')
                    handle.flush();os.fsync(handle.fileno())
        rows=_ledger(output,config,families,tokenizer,inherited=inherited)
        limit=len(families) if max_families is None else min(len(families),max(0,max_families-len(inherited)))
        require(max_families is None or max_families >= len(inherited)+len(rows), 'Cannot shrink inspected prefix')
        model=device=None
        try:
            from transformers import GenerationConfig
            generation=GenerationConfig.from_dict(config['generation'])
            passes=sum(r['triage_selected'] for r in inherited+rows)
            for index in range(len(rows),limit):
                if passes==TARGET:break
                source=families[index]
                if not source['tiny_preflight']['passed']:
                    state='tiny_preflight_reject'; evidence=None; judgment=quality._unknown(state)
                elif len(source['input_ids'])>4096:
                    state='quality_input_cap'; evidence=None; judgment=quality._unknown(state)
                else:
                    if model is None:
                        device=quality._device(judge_profile=PROFILE); train._reset_seed(device)
                        model=quality._new_model(snapshot,device,generation,judge_profile=PROFILE)
                    generated=runtime.generate_batch(model,tokenizer,[source['input_ids']],generation_config=generation,device=device)
                    require(len(generated)==1,'Generation omitted whole-family evidence')
                    evidence=generated[0]; state='generated'
                    judgment=quality.parse_judgment(evidence['response'],finish_reason=evidence['finish_reason'])
                row=dict(family_index=len(inherited)+index,source_index=source['source_index'],family_id=source['family_id'],
                    config_sha256=_hash_file(output/'run-config.json'),input_sha256=source['input_sha256'],state=state,evidence=evidence,
                    judgment=judgment,triage_selected=judgment['family_pass'],rejection_reason=_rejection(state,judgment))
                _ledger(output,config,families,tokenizer,records=[row],start_index=index,inherited=inherited)
                with (output/'families.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(_json(row)+'\n');handle.flush();os.fsync(handle.fileno())
                rows.append(row); passes+=row['triage_selected']
                if len(inherited)+len(rows)==PILOT:_save(output,config,inherited+rows)
            return _save(output,config,inherited+rows)
        except BaseException as exc:
            _save(output,config,inherited+rows,failed=type(exc).__name__)
            raise
        finally:
            model=None;gc.collect()
            if device=='mps':
                import torch
                torch.mps.empty_cache()


def load_scores(output: Path,prepared: Path) -> dict:
    """CPU-only immutable reader, including original-config inherited records."""
    output,prepared=Path(output),Path(prepared)
    require(not (output/'.writer.lock').exists(),'Triage writer lock exists')
    config=_read(output/'run-config.json')
    readiness_path=_reference(output,config['bindings']['readiness'])
    extension=config.get('extension_of'); extension_path=None
    if extension is not None:
        require(type(extension) is dict and set(extension)=={'path','sha256','prepared','inherited_rows_sha256','config_sha256'}, 'Invalid extension binding')
        extension_path=_reference(output,{'path':extension['path'],'sha256':extension['sha256']}).parent
        _reference(output,extension['prepared'])
    expected,families,tokenizer,_,inherited,inherited_families=_context(prepared,readiness_path,output,extension_of=extension_path)
    _validate_config(output,config,expected)
    require(_same(_rows(output/'inputs.jsonl'),families),'Frozen triage input/prompt/token mismatch')
    rows=inherited+_ledger(output,config,families,tokenizer,inherited=inherited)
    report,manifest=_verify_manifest(output,config,rows)
    return dict(config=config,rows=rows,families=inherited_families+families,report=report,manifest=manifest,
        selected_family_ids=[r['family_id'] for r in rows if r['triage_selected']],
        bindings=config['bindings'],inherited_rows=len(inherited))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    ready=commands.add_parser('readiness');ready.add_argument('--triage-readiness',type=Path,required=True)
    for name in ('score','verify'):
        command=commands.add_parser(name);command.add_argument('--prepared',type=Path,required=True);command.add_argument('--output',type=Path,required=True)
        if name=='score':
            command.add_argument('--triage-readiness',type=Path,required=True);command.add_argument('--max-families',type=int);command.add_argument('--extension-of',type=Path)
    args=parser.parse_args(argv)
    if args.command=='readiness':
        value=load_readiness(args.triage_readiness)
        report=dict(sha256=value['sha256'],calibration_qualification='not_established',allowed_use='candidate_triage_only',exclusions=value['exclusions'])
    elif args.command=='score':report=score(args.prepared,args.triage_readiness,args.output,args.max_families,extension_of=args.extension_of)
    else:report=load_scores(args.output,args.prepared)['report']
    print(_json(report));return report


if __name__=='__main__':main()
