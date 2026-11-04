"""Bounded, CPU-only independent source audit after provisional Tiny Aya triage.

Integrity checks locate review evidence; they do not certify the reviewer's factual
claims. Root must inspect semantic support. Qwen qualification and human QC remain
unestablished/pending, including after local pilot data acceptance.
"""
import argparse
from collections import Counter
import copy
import json
import tempfile
import os
from pathlib import Path
import re

from . import prepare, guard, curate
from .guard import _hash_file, _digest, _json
from .quality import require, _read, _rows, _same

LANGUAGES = ('en', 'ko', 'zh')
LABELS = ('usable', 'incorrect', 'ambiguous', 'unclear')
SALT = 'tiny-aya-quality-audit-v1:'
PENDING = 'provisional_triage_candidates_pending_safety_and_independent_audit'
ACCEPTED = ('accepted_initial', 'accepted_after_repair')
PACK_FILES = {'pack.json', 'rows.jsonl', 'annotations.template.json'}
CLOSED_FILES = PACK_FILES | {'manifest.json', 'annotations.json', 'report.json', 'review-manifest.json'}
POLICY = dict(families=30, languages=list(LANGUAGES), fail_if_incorrect_per_language_gt=3, max_repairs=1)


def _path(path, root=None):
    raw=Path(path).absolute()
    root=Path(root or curate._evidence_root()).resolve()
    # macOS /var aliases the configured /private/var root. Permit that root
    # alias, while rejecting every symlink within the actual evidence tree.
    for node in (raw,*raw.parents):
        if node.resolve()==root: break
        require(not node.is_symlink(), 'Symlink evidence is forbidden')
    resolved=raw.resolve()
    require(resolved.is_relative_to(root), 'Evidence path outside configured root')
    return resolved


def _safe(path, root=None):
    path=_path(path,root)
    require(path.is_file(), 'Missing evidence file')
    return path


def _write(path,value):
    """Atomic fresh creation, including race-safe refusal to overwrite evidence."""
    path=Path(path)
    descriptor,temporary=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as handle:
            handle.write(_json(value)+'\n');handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _ref(base, path):
    path = _safe(path)
    return dict(path=os.path.relpath(path, Path(base).absolute()), sha256=_hash_file(path))


def _resolve(base, ref, root=None):
    require(type(ref) is dict and set(ref)=={'path','sha256'} and type(ref['path']) is str and
            bool(ref['path']) and not Path(ref['path']).is_absolute() and type(ref['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}',ref['sha256']), 'Invalid confined evidence reference')
    path = _safe(Path(os.path.abspath(Path(base)/ref['path'])), root)
    require(_hash_file(path)==ref['sha256'], 'Evidence hash mismatch')
    return path


def _inventory(directory):
    directory = Path(directory).absolute()
    require(not directory.is_symlink(), 'Symlink directory forbidden')
    paths = list(directory.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Symlink evidence forbidden')
    return {p.relative_to(directory).as_posix():_hash_file(p) for p in paths if p.is_file()}


def _collect(seeds, root):
    """Copy only the bounded file-reference closure of the existing artifact types."""
    found = set()
    def add(path):
        path = _safe(path, root)
        if path in found: return
        found.add(path)
        require(path.suffix in ('.json','.jsonl','.py'), 'Unexpected evidence file type')
        if path.suffix != '.json': return
        value = _read(path)
        if type(value) is not dict: return
        def refs(node):
            if type(node) is dict:
                if set(node)=={'path','sha256'}: add(_resolve(path.parent,node,root))
                else:
                    for item in node.values(): refs(item)
            elif type(node) is list:
                for item in node: refs(item)
        refs(value)
        if type(value.get('files')) is dict:
            for name, entry in value['files'].items():
                digest = entry.get('sha256') if type(entry) is dict else entry
                add(_resolve(path.parent,dict(path=name,sha256=digest),root))
        if 'config_sha256' in value and 'config' in value:
            add(_resolve(path.parent,dict(path='config.json',sha256=value['config_sha256']),root))
        if 'prepared_path' in value:
            relative=value['prepared_path']
            require(type(relative) is str and not Path(relative).is_absolute(), 'Invalid prepared reference')
            add(Path(os.path.abspath(path.parent/relative)))
    for seed in seeds: add(seed)
    return found


def _copy_evidence(destination, anchors):
    root=curate._evidence_root()
    files=_collect(anchors.values(),root)
    destination.mkdir(parents=True,exist_ok=False)
    for source in sorted(files):
        target=destination/'tree'/source.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    copied={key:destination/'tree'/path.absolute().relative_to(root) for key,path in anchors.items()}
    _write(destination/'manifest.json',dict(schema_version=1,files=_inventory(destination),
        anchors={key:_ref(destination,path) for key,path in copied.items()}))
    return copied


def _evidence(prepared, manifest):
    q=manifest.get('quality_curation')
    require(type(q) is dict, 'Missing quality curation metadata')
    evidence=_resolve(prepared,q.get('quality_evidence'))
    require(evidence==prepared/'quality-evidence'/'manifest.json', 'Wrong quality evidence location')
    value=_read(evidence)
    require(type(value) is dict and set(value)=={'schema_version','files','anchors'} and type(value['schema_version']) is int and value['schema_version']==1,
            'Invalid quality evidence manifest')
    actual=_inventory(evidence.parent);actual.pop('manifest.json',None)
    require(actual==value['files'], 'Quality evidence inventory mismatch')
    anchors={k:_resolve(evidence.parent,v,evidence.parent/'tree') for k,v in value['anchors'].items()}
    expected={'pool','scores','readiness'} | ({'original_review','original_prepared','original_screened'} if q.get('repair_audit') is not None else set())
    require(set(anchors)==expected, 'Wrong quality evidence anchors')
    closure=_collect(anchors.values(),evidence.parent/'tree')
    require({p.relative_to(evidence.parent).as_posix() for p in closure}==set(actual), 'Omitted/mixed quality evidence')
    return anchors,evidence


def _selected(pool, scores):
    loaded=curate.load_scores(scores,pool)
    ids=loaded['selected_family_ids']
    require(loaded['report']['status']=='target_reached' and len(ids)==900 and len(set(ids))==900,
            'Completed first900 triage selections required')
    families={f['family_id']:f['rows'] for f in loaded['families']}
    require(len(families)==len(loaded['families']) and set(ids)<=set(families), 'Incomplete selected family source')
    require(not set(ids)&set(loaded['bindings']['exclusions']), 'Calibration family selected')
    return loaded,ids,families


def _quality_metadata(output,anchors,evidence,ids,repair):
    return dict(schema_version=1,status=PENDING,calibration_qualification='not_established',human_review='pending',
        parent_pool=_ref(output,anchors['pool']),scores=_ref(output,anchors['scores']),
        triage_readiness=_ref(output,anchors['readiness']),selected_family_ids=ids,
        source_statistics_scope='parent_pool_before_triage',quality_evidence=_ref(output,evidence),repair_audit=repair)


def freeze(prepared: Path, scores: Path, output: Path, *, repair_audit: Path|None=None) -> dict:
    prepared,scores,output=map(lambda p:_path(p),(prepared,scores,output))
    parent=prepare.verify_bundle(prepared)
    loaded,ids,families=_selected(prepared,scores)
    ready=curate._reference(scores,loaded['bindings']['readiness'])
    anchors=dict(pool=prepared/'manifest.json',scores=scores/'manifest.json',readiness=ready)
    excluded=[]
    if repair_audit is not None:
        old=_closed_review(_path(repair_audit))
        require(old['report']['mode']=='initial' and old['report']['status']=='repair_required', 'Only one initial repair-required review permits repair')
        old_prepared=old['prepared'];old_screened=old['screened']
        old_manifest=old['context']['manifest']
        q=old_manifest['quality_curation']
        require(q['repair_audit'] is None and q['selected_family_ids']==ids and
                q['parent_pool']['sha256']==_hash_file(prepared/'manifest.json') and
                q['scores']['sha256']==_hash_file(scores/'manifest.json'), 'Repair triage candidate identity changed')
        excluded=[identity for identity in ids if identity in old['report']['confirmed_bad_family_ids']]
        require(1<=len(excluded)<=9, 'Invalid repair union')
        anchors.update(original_review=_path(repair_audit)/'review-manifest.json',
            original_prepared=old_prepared/'manifest.json',original_screened=old_screened/'manifest.json')
    require(not output.exists(), 'Freeze output already exists')
    require(output.is_relative_to(curate._evidence_root()) and not any(output.is_relative_to(p.parent) for p in anchors.values()),
            'Overlapping/outside freeze output')
    output.mkdir(parents=True)
    copied=_copy_evidence(output/'quality-evidence',anchors)
    repair=None
    if excluded:
        repair=dict(generation=1,original_review=_ref(output,copied['original_review']),
            original_prepared=_ref(output,copied['original_prepared']),original_screened=_ref(output,copied['original_screened']),
            excluded_family_ids=excluded)
    kept=[i for i in ids if i not in excluded]
    rows=[row for identity in kept for row in families[identity]]
    for name in curate.AUXILIARY_FILES: (output/name).write_bytes((prepared/name).read_bytes())
    (output/'candidates.jsonl').write_text(''.join(_json(r)+'\n' for r in rows),encoding='utf-8')
    manifest=copy.deepcopy(parent)
    manifest['candidate_families']=len(kept);manifest['counts']['candidates']=len(rows)
    manifest['files']['candidates.jsonl']=dict(sha256=_hash_file(output/'candidates.jsonl'),rows=len(rows))
    manifest['candidate_statistics']['selected_families']=len(kept)
    manifest['candidate_statistics']['selected_changed_instructions']=sum(r['provenance']['instruction_changed'] for r in rows)
    manifest['quality_curation']=_quality_metadata(output,copied,output/'quality-evidence'/'manifest.json',ids,repair)
    _write(output/'manifest.json',manifest)
    return verify_bundle(output)


def verify_bundle(prepared: Path) -> dict:
    prepared=_path(prepared)
    require({p.name for p in prepared.iterdir()}==set(prepare.DATA_FILES)|{'config.json','manifest.json','quality-evidence'},
            'Unexpected prepared curation inventory')
    for name in (*prepare.DATA_FILES,'config.json','manifest.json'): _safe(prepared/name)
    manifest=prepare.verify_bundle(prepared)
    anchors,evidence=_evidence(prepared,manifest)
    pool,scores=anchors['pool'].parent,anchors['scores'].parent
    loaded,ids,families=_selected(pool,scores)
    require(anchors['readiness']==curate._reference(scores,loaded['bindings']['readiness']), 'Mixed readiness evidence')
    repair=manifest['quality_curation']['repair_audit'];excluded=[]
    if repair is not None:
        require(type(repair) is dict and set(repair)=={'generation','original_review','original_prepared','original_screened','excluded_family_ids'} and
                type(repair['generation']) is int and repair['generation']==1, 'Invalid single repair metadata')
        for key in ('original_review','original_prepared','original_screened'):
            require(_resolve(prepared,repair[key])==anchors[key], 'Mixed repair evidence')
        original=_closed_review(anchors['original_review'].parent)
        require(original['report']['mode']=='initial' and original['report']['status']=='repair_required', 'Invalid original repair result')
        require(original['prepared']==anchors['original_prepared'].parent and original['screened']==anchors['original_screened'].parent,
                'Repair duplicated manifests disagree')
        oq=original['context']['manifest']['quality_curation']
        require(oq['repair_audit'] is None and oq['selected_family_ids']==ids and oq['parent_pool']['sha256']==_hash_file(anchors['pool']) and
                oq['scores']['sha256']==_hash_file(anchors['scores']), 'Repair source identity changed')
        excluded=[i for i in ids if i in original['report']['confirmed_bad_family_ids']]
        require(repair['excluded_family_ids']==excluded and 1<=len(excluded)<=9, 'Repair must remove exact confirmed-bad union')
    expected=_quality_metadata(prepared,anchors,evidence,ids,repair)
    require(_same(manifest['quality_curation'],expected), 'Invalid quality curation authority/bindings')
    rows=[r for i in ids if i not in excluded for r in families[i]]
    parent=prepare.verify_bundle(pool)
    rebuilt=copy.deepcopy(parent)
    rebuilt['candidate_families']=len(ids)-len(excluded);rebuilt['counts']['candidates']=len(rows)
    rebuilt['files']['candidates.jsonl']=dict(sha256=_hash_file(prepared/'candidates.jsonl'),rows=len(rows))
    rebuilt['candidate_statistics']['selected_families']=len(ids)-len(excluded)
    rebuilt['candidate_statistics']['selected_changed_instructions']=sum(r['provenance']['instruction_changed'] for r in rows)
    rebuilt['quality_curation']=expected
    require(_same(manifest,rebuilt) and _rows(prepared/'candidates.jsonl')==rows and
            all((prepared/n).read_bytes()==(pool/n).read_bytes() for n in curate.AUXILIARY_FILES), 'Selected source rows/statistics/auxiliary changed')
    return manifest


def _partitions(prepared,screened):
    manifest=verify_bundle(prepared)
    for name in ('manifest.json','run-config.json','judgments.jsonl','report.json',*guard.SPLITS):_safe(screened/name)
    safety=guard.load_screening(screened,prepared)
    candidates={r['id']:r for r in _rows(prepared/'candidates.jsonl')}
    ids=[]
    for key,count in (('train',600),('eval_task',100)):
        rows=safety[key]
        family_ids=[]
        require(len(rows)==count*3,'Exact600 train /100 eval required')
        for offset in range(0,len(rows),3):
            group=rows[offset:offset+3];identity=group[0]['family_id'];family_ids.append(identity)
            require([r['lang'] for r in group]==list(LANGUAGES) and all(r['family_id']==identity for r in group), 'Incomplete safety family')
            require(all(r['id'] in candidates and all(r.get(k)==v for k,v in candidates[r['id']].items()) for r in group), 'Safety source rows changed')
        require(len(set(family_ids))==count,'Duplicate safety families')
        ids.append(family_ids)
    train_ids,eval_ids=ids
    require(not set(train_ids)&set(eval_ids),'Train/eval overlap')
    return manifest,safety,train_ids,eval_ids


def _audit_context(prepared,screened):
    manifest,safety,train_ids,eval_ids=_partitions(prepared,screened)
    repair=manifest['quality_curation']['repair_audit'];previous=None;excluded=[]
    if repair is None:
        mode='initial';selection=dict(rule='hash_first30',salt=SALT)
        import hashlib
        selected=sorted(train_ids,key=lambda i:(hashlib.sha256((SALT+i).encode()).hexdigest(),i))[:30]
    else:
        mode='replacements';selection=dict(rule='newly_introduced_train_in_guard_order',salt=None)
        previous=_resolve(prepared,repair['original_review'])
        original=_closed_review(previous.parent)
        old=original['context']['safety'];old_train=original['context']['train'];old_eval=original['context']['eval']
        excluded=repair['excluded_family_ids']
        require(set(old_train)-set(train_ids)==set(excluded) and set(excluded)<=set(old_train) and
                not set(excluded)&(set(train_ids)|set(eval_ids)), 'Unexpected retained training turnover or surviving known error')
        require(eval_ids==old_eval and safety['eval_task']==old['eval_task'], 'Repair evaluation rows changed')
        retained=set(old_train)&set(train_ids)
        require([r for r in safety['train'] if r['family_id'] in retained]==[r for r in old['train'] if r['family_id'] in retained],
                'Retained train rows/order changed')
        selected=[i for i in train_ids if i not in old_train]
        require(len(selected)==len(excluded) and 1<=len(selected)<=9,'Wrong exact replacement set')
    original_rows={r['id']:r for r in _rows(prepared/'candidates.jsonl')}
    rows=[{k:original_rows[f'{i}:{lang}'][k] for k in ('id','family_id','lang','instruction','response')}
          for i in selected for lang in LANGUAGES]
    return dict(mode=mode,selection=selection,selected=selected,train=train_ids,eval=eval_ids,rows=rows,
                previous=previous,excluded=excluded,evidence=manifest['quality_curation']['quality_evidence']['sha256'],
                manifest=manifest,safety=safety)


def _pack(output,prepared,screened,context):
    return dict(schema_version=1,status='frozen_pending_review',mode=context['mode'],human_review='pending',
        prepared_manifest=_ref(output,prepared/'manifest.json'),screened_manifest=_ref(output,screened/'manifest.json'),
        triage_evidence_sha256=context['evidence'],train_family_ids=context['train'],eval_family_ids=context['eval'],
        selected_family_ids=context['selected'],selection=context['selection'],
        previous_review=None if context['previous'] is None else _ref(output,context['previous']),
        excluded_family_ids=context['excluded'],rows_sha256=_hash_file(output/'rows.jsonl'),expected_rows=len(context['rows']))


def _template(rows):
    return dict(schema_version=1,pack_manifest_sha256=None,status='pending',
        review_kind='independent_automated_source_review',human_review='pending',
        reviewer=dict(name=None,model_or_version=None,method=None),
        annotations=[{k:r[k] for k in ('id','family_id','lang')} | dict(quality=None,reason=None,source_evidence=[]) for r in rows])


def freeze_audit(prepared: Path, screened: Path, output: Path) -> dict:
    prepared,screened,output=map(lambda p:_path(p),(prepared,screened,output))
    context=_audit_context(prepared,screened)
    require(not output.exists() and output.is_relative_to(curate._evidence_root()),'Audit output exists or is outside evidence root')
    output.mkdir(parents=True)
    (output/'rows.jsonl').write_text(''.join(_json(r)+'\n' for r in context['rows']),encoding='utf-8')
    _write(output/'pack.json',_pack(output,prepared,screened,context))
    # The approved pending null avoids a manifest/template hash cycle.
    template=_template(context['rows'])
    _write(output/'annotations.template.json',template)
    _write(output/'manifest.json',dict(schema_version=1,files={n:dict(sha256=_hash_file(output/n),rows=len(context['rows']) if n=='rows.jsonl' else 1) for n in sorted(PACK_FILES)}))
    return dict(status='awaiting_independent_review',mode=context['mode'],family_ids=context['selected'],row_count=len(context['rows']),
                paths={n:output/n for n in sorted(PACK_FILES|{'manifest.json'})})


def _review(audit,prepared=None,screened=None):
    audit=Path(audit).absolute()
    inventory=set(_inventory(audit))
    require(inventory in (PACK_FILES|{'manifest.json','annotations.json'},CLOSED_FILES), 'Incomplete/mixed audit inventory')
    manifest=_read(_safe(audit/'manifest.json'))
    require(_same(manifest,dict(schema_version=1,files={n:dict(sha256=_hash_file(audit/n),rows=len(_rows(audit/'rows.jsonl')) if n=='rows.jsonl' else 1) for n in sorted(PACK_FILES)})), 'Audit pack inventory/hash mismatch')
    pack=_read(audit/'pack.json')
    require(type(pack) is dict,'Invalid audit pack metadata')
    bound_prepared=_resolve(audit,pack.get('prepared_manifest')).parent
    bound_screened=_resolve(audit,pack.get('screened_manifest')).parent
    require((prepared is None or prepared==bound_prepared) and (screened is None or screened==bound_screened), 'Wrong audit prepared/screened binding')
    context=_audit_context(bound_prepared,bound_screened)
    require(_rows(audit/'rows.jsonl')==context['rows'] and _same(pack,_pack(audit,bound_prepared,bound_screened,context)),'Audit source/selection/lineage mismatch')
    expected_template=_template(context['rows'])
    require(_same(_read(audit/'annotations.template.json'),expected_template),'Pending template changed')
    annotations=_read(audit/'annotations.json')
    require(type(annotations) is dict and set(annotations)==set(expected_template), 'Invalid annotation schema')
    # A null template binding avoids a hash cycle; completed annotations bind the
    # now-frozen immutable manifest, and must supply that exact digest.
    require(type(annotations['schema_version']) is int and annotations['schema_version']==1 and annotations['pack_manifest_sha256']==_hash_file(audit/'manifest.json') and
            annotations['status']=='complete' and annotations['review_kind']=='independent_automated_source_review' and
            annotations['human_review']=='pending', 'Independent completed review required')
    reviewer=annotations['reviewer']
    require(type(reviewer) is dict and set(reviewer)=={'name','model_or_version','method'} and
            all(type(v) is str and v.strip() for v in reviewer.values()),'Missing independent reviewer provenance')
    rows=annotations['annotations']
    require(type(rows) is list and len(rows)==len(context['rows']),'Incomplete independent audit rows')
    counts={lang:dict(denominator=len(context['selected']),**{label:0 for label in LABELS}) for lang in LANGUAGES}
    bad=set()
    for annotation,source in zip(rows,context['rows']):
        require(type(annotation) is dict and set(annotation)=={'id','family_id','lang','quality','reason','source_evidence'} and
                all(annotation[k]==source[k] for k in ('id','family_id','lang')) and annotation['quality'] in LABELS and
                type(annotation['reason']) is str and annotation['reason'].strip(),'Invalid independent label/reason/order')
        quotes=annotation['source_evidence']
        require(type(quotes) is list,'Invalid source evidence')
        for quote in quotes:
            require(type(quote) is dict and set(quote)=={'field','quote'} and quote['field'] in ('instruction','response') and
                    type(quote['quote']) is str and quote['quote'].strip() and quote['quote'] in source[quote['field']], 'Nonliteral/wrong-field source evidence')
        if annotation['quality']=='incorrect':
            require(bool(quotes),'Incorrect judgment needs source evidence');bad.add(source['family_id'])
        counts[source['lang']][annotation['quality']]+=1
    exceeded=[lang for lang in LANGUAGES if counts[lang]['incorrect']>3]
    status=('failed_threshold' if exceeded else 'repair_required' if bad else 'accepted_initial') if context['mode']=='initial' else (
            'failed_replacement_review' if bad else 'accepted_after_repair')
    report=dict(schema_version=1,mode=context['mode'],status=status,pack_manifest_sha256=_hash_file(audit/'manifest.json'),
        annotations_sha256=_hash_file(audit/'annotations.json'),human_review='pending',calibration_qualification='not_established',
        reviewed_families=len(context['selected']),reviewed_rows=len(context['rows']),per_language_counts=counts,
        confirmed_bad_family_ids=[i for i in context['selected'] if i in bad],threshold_exceeded_languages=exceeded,
        previous_review_sha256=None if context['previous'] is None else _hash_file(context['previous']),training_approved=status in ACCEPTED)
    return dict(pack=pack,report=report,prepared=bound_prepared,screened=bound_screened,context=context)


def _review_manifest(audit,report):
    return dict(schema_version=1,status=report['status'],files={n:_ref(audit,audit/n) for n in ('manifest.json','annotations.json','report.json')})


def _closed_review(audit):
    result=_review(audit)
    require(set(_inventory(audit))==CLOSED_FILES and _same(_read(audit/'report.json'),result['report']) and
            _same(_read(audit/'review-manifest.json'),_review_manifest(audit,result['report'])), 'Closed independent review changed')
    return result


def _sidecar(prepared,screened,audit,review,original=None):
    require(review['report']['status'] in ACCEPTED,'Independent review did not accept data')
    context=review['context'];manifest=context['manifest']
    evidence=_resolve(prepared,manifest['quality_curation']['quality_evidence'])
    return dict(schema_version=1,status='provisional_independent_audit_acceptance',
        acceptance_basis='independent_blinded_training_source_audit',calibration_qualification='not_established',human_review='pending',
        prepared_manifest=_ref(screened,prepared/'manifest.json'),screened_manifest=_ref(screened,screened/'manifest.json'),
        quality_evidence=_ref(screened,evidence),triage_readiness=manifest['quality_curation']['triage_readiness']['sha256'],
        audit_policy=POLICY,review=_ref(screened,audit/'review-manifest.json'),
        original_review=None if original is None else _ref(screened,original/'review-manifest.json'),
        train_family_ids_sha256=_digest(context['train']),eval_family_ids_sha256=_digest(context['eval']),
        excluded_family_ids=context['excluded'])


def accept(prepared: Path, screened: Path, audit: Path, *, previous: Path|None=None) -> dict:
    prepared,screened,audit=map(lambda p:_path(p),(prepared,screened,audit))
    result=_review(audit,prepared,screened)
    context=result['context']
    if context['mode']=='initial': require(previous is None,'Initial review forbids previous lineage')
    else:
        require(previous is not None and _hash_file(_safe(Path(previous)/'review-manifest.json'))==_hash_file(context['previous']), 'Replacement requires exact original review')
        original=_closed_review(_path(previous))
        require(original['report']['mode']=='initial' and original['report']['status']=='repair_required',
                'Previous directory must contain the complete original repair-required review')
    if (audit/'review-manifest.json').exists(): _closed_review(audit)
    else:
        require(not (audit/'report.json').exists(),'Partial review closure; refuse overwrite')
        _write(audit/'report.json',result['report'])
        _write(audit/'review-manifest.json',_review_manifest(audit,result['report']))
    sidecar_path=None;sidecar_hash=None
    if result['report']['status'] in ACCEPTED:
        sidecar_path=screened/'quality-acceptance.json'
        if sidecar_path.exists():
            existing=verify_acceptance(prepared,screened)
            require(existing['sidecar']['review']['sha256']==_hash_file(audit/'review-manifest.json'),
                    'Existing acceptance belongs to a different completed review')
        else:
            destination=screened/'quality-audit'
            anchors=dict(review=audit/'review-manifest.json')
            if context['previous'] is not None:anchors['original_review']=context['previous']
            copied=_copy_evidence(destination,anchors)
            copied_audit=copied['review'].parent
            original=copied.get('original_review')
            copied_review=_closed_review(copied_audit)
            value=_sidecar(prepared,screened,copied_audit,copied_review,None if original is None else original.parent)
            _write(sidecar_path,value)
            verify_acceptance(prepared,screened)
        sidecar_hash=_hash_file(sidecar_path)
    return dict(result['report'],acceptance_sidecar_path=sidecar_path,acceptance_sidecar_sha256=sidecar_hash)


def verify_acceptance(prepared: Path, screened: Path) -> dict:
    prepared,screened=_path(prepared),_path(screened)
    _partitions(prepared,screened)
    path=_safe(screened/'quality-acceptance.json');value=_read(path)
    review_path=_resolve(screened,value.get('review'),screened/'quality-audit'/'tree')
    original=value.get('original_review');original_path=None if original is None else _resolve(screened,original,screened/'quality-audit'/'tree')
    result=_closed_review(review_path.parent)
    require(_hash_file(result['prepared']/'manifest.json')==_hash_file(prepared/'manifest.json') and
            _hash_file(result['screened']/'manifest.json')==_hash_file(screened/'manifest.json'), 'Sidecar split binding mismatch')
    if result['context']['mode']=='initial':require(original_path is None,'Unexpected original audit')
    else:require(original_path==result['context']['previous'],'Wrong original repair audit')
    expected=_sidecar(prepared,screened,review_path.parent,result,None if original_path is None else original_path.parent)
    require(_same(value,expected),'Acceptance status/authority/evidence changed')
    evidence=_read(screened/'quality-audit'/'manifest.json')
    require(type(evidence) is dict and set(evidence)=={'schema_version','files','anchors'} and type(evidence['schema_version']) is int and evidence['schema_version']==1,'Invalid audit evidence manifest')
    actual=_inventory(screened/'quality-audit');actual.pop('manifest.json',None)
    anchors={k:_resolve(screened/'quality-audit',v,screened/'quality-audit'/'tree') for k,v in evidence['anchors'].items()}
    require(anchors==dict(review=review_path,**({} if original_path is None else dict(original_review=original_path))) and actual==evidence['files'] and
            {p.relative_to(screened/'quality-audit').as_posix() for p in _collect(anchors.values(),screened/'quality-audit'/'tree')}==set(actual),'Mixed/omitted audit evidence')
    bindings=dict(quality_acceptance_sha256=_hash_file(path),quality_evidence_sha256=value['quality_evidence']['sha256'])
    return dict(bindings=bindings,**bindings,report=result['report'],sidecar=value)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('freeze','freeze-audit','accept','verify'):
        command=commands.add_parser(name);command.add_argument('--prepared',type=Path,required=True)
        if name=='freeze':
            command.add_argument('--scores',type=Path,required=True);command.add_argument('--repair-audit',type=Path)
        else:command.add_argument('--screened',type=Path,required=name!='verify')
        if name in ('freeze','freeze-audit'):command.add_argument('--output',type=Path,required=True)
        if name=='accept':command.add_argument('--audit',type=Path,required=True);command.add_argument('--previous',type=Path)
    args=parser.parse_args(argv)
    if args.command=='freeze':result=freeze(args.prepared,args.scores,args.output,repair_audit=args.repair_audit)
    elif args.command=='freeze-audit':result=freeze_audit(args.prepared,args.screened,args.output)
    elif args.command=='accept':result=accept(args.prepared,args.screened,args.audit,previous=args.previous)
    else:result=verify_acceptance(args.prepared,args.screened) if args.screened else verify_bundle(args.prepared)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False,default=str));return result


if __name__=='__main__': main()
