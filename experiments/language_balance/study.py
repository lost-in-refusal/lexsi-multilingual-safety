"""Prepare a training-only reviewed pool and two immutable language schedules.

Original references remain unchanged. Corrected targets are explicitly derived
and enter only after independent usable+benign consensus. Holdouts stay opaque.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random

from experiments.multilingual_repair import train as v1

require, sha_bytes, write_json = v1.require, v1.sha_bytes, v1.write_json
QUOTAS = {'control': {'en': 450, 'ko': 100, 'zh': 450},
          'balanced': {'en': 333, 'ko': 334, 'zh': 333}}
FILES = {'inputs.jsonl', 'training-config.json', 'annotations.json', 'report.json'}


def read_source(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Missing/symlinked provenance source: {path.name}')
    data = path.read_bytes()
    return {'path': str(path.resolve().relative_to(v1.ROOT)) if path.resolve().is_relative_to(v1.ROOT) else path.name, 'sha256': sha_bytes(data), 'utf8': data.decode('utf-8')}


def exact_index(rows, ids, description):
    require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), f'{description}: expected array')
    by_id = {r.get('id'): r for r in rows}
    require(len(by_id) == len(rows) and set(by_id) == set(ids), f'{description}: duplicate/wrong source IDs or incomplete coverage')
    return by_id


def build_schedules(pool, alignment):
    require(isinstance(pool, list) and bool(pool), 'Empty pool')
    by_id = {r['id']: r for r in pool}
    require(len(by_id) == len(pool) and all(r['split'] == 'train' for r in pool), 'Duplicate/nontraining pool IDs')
    draws = {}
    for offset, language in enumerate(('en', 'ko', 'zh')):
        ids = sorted(r['id'] for r in pool if r['language'] == language)
        require(bool(ids), 'Every language requires a nonempty pool')
        rng = random.Random(42 + offset)
        draws[language] = []
        while len(draws[language]) < max(q[language] for q in QUOTAS.values()):
            cycle = ids[:]; rng.shuffle(cycle); draws[language].extend(cycle)
    schedules = {}
    for arm, quota in QUOTAS.items():
        languages = [language for language in ('en', 'ko', 'zh') for _ in range(quota[language])]
        random.Random(42).shuffle(languages)
        positions = Counter(); order = []
        for language in languages:
            order.append(draws[language][positions[language]]); positions[language] += 1
        schedules[arm] = order
    alignment_ids = [r['id'] for r in alignment]
    require(len(set(alignment_ids)) == len(alignment_ids) == 72, 'Expected unchanged alignment72')
    random.Random(43).shuffle(alignment_ids)
    return schedules, [alignment_ids[i % 72] for i in range(1000)]


def validate_review(rows, ids, description):
    by_id = exact_index(rows, ids, description)
    for row in by_id.values():
        require(set(row) == {'id', 'quality', 'safety', 'notes'}, f'{description}: wrong review fields')
        require(row['quality'] in ('usable', 'incorrect', 'ambiguous', 'unclear')
                and row['safety'] in ('benign', 'unsafe', 'unclear')
                and isinstance(row['notes'], str) and bool(row['notes'].strip()), f'{description}: invalid review')
    return by_id


def derive_pool(bundle, sources, tokenizer=None):
    """Replay all identity/consensus gates from authenticated original source bytes."""
    parse = lambda label: json.loads(sources[label]['utf8'])
    train = bundle['train']; by_id = {r['id']: r for r in train}
    audits = []
    for label in ('audit-a', 'audit-b'):
        annotations, report = parse(label + '-annotations'), parse(label + '-report')
        require(((label=='audit-a' and report.get('audit_type')=='provisional_automated_training_data_audit')
                 or (label=='audit-b' and report.get('audit_kind')=='independent provisional Korean training-data audit'
                     and report.get('review_status')=='model-generated labels; not human certification'))
                and report.get('annotations_sha256') == sources[label + '-annotations']['sha256']
                and report.get('source_train_sha256') == sha_bytes(bundle['blobs']['train.jsonl']), 'Original audit binding mismatch')
        audit = validate_review(annotations, {r['id'] for r in train if r['language']=='ko'}, label)
        for row in audit.values():
            require(row.get('quality') in ('usable', 'incorrect', 'ambiguous', 'unclear')
                    and row.get('safety') in ('benign', 'unsafe', 'unclear'), 'Invalid original audit labels')
        audits.append(audit)
    korean = {r['id'] for r in train if r['language'] == 'ko'}
    core = {i for i in korean if all(a[i]['quality'] == 'usable'
                                    and a[i]['safety'] == 'benign' for a in audits)}
    require(len(core) == 34, 'Expected original34 Korean both-accepted core')
    gate={i for i in korean if all(a[i]['quality']=='incorrect' for a in audits) or any(a[i]['safety']=='unsafe' for a in audits)}
    require(len(gate)==18, 'Expected exact18 repair source IDs')
    repairs = exact_index(parse('repairs-annotations'), gate, 'repairs')
    repair_inputs=exact_index([json.loads(line) for line in sources['repairs-inputs']['utf8'].splitlines() if line.strip()],gate,'repair input pack')
    for identity,row in repair_inputs.items():
        require(set(row)=={'id','source_id','language','request','system','reference','audit_notes'}
                and all(row[k]==by_id[identity][k] for k in ('id','source_id','language','request','system','reference')), 'Repair source fields changed')
    repair_report=parse('repairs-report')
    require(repair_report.get('status')=='completed_provisional_target_repair' and repair_report.get('records')==18
            and repair_report.get('inputs_sha256')==sources['repairs-inputs']['sha256']
            and repair_report.get('annotations_sha256')==sources['repairs-annotations']['sha256']
            and repair_report.get('source_train_sha256')==sha_bytes(bundle['blobs']['train.jsonl']), 'Repair source report binding mismatch')
    for repair in repairs.values():
        require(set(repair) == {'id','status','reference','rationale','sources'}, 'Wrong repair fields')
        require(repair['status'] in ('repaired','excluded') and isinstance(repair['reference'],str)
                and isinstance(repair['rationale'],str) and bool(repair['rationale'].strip())
                and isinstance(repair['sources'],list) and all(isinstance(s,str) and s.startswith(('https://','http://'))
                                                           for s in repair['sources']), 'Invalid repair')
        require(bool(repair['reference'].strip()) if repair['status']=='repaired' else repair['reference']=='', 'Invalid repaired/excluded reference')
    proposed = {i for i,r in repairs.items() if r['status']=='repaired'}
    review_sets=[]
    require(sources['repair-review-a-inputs']['utf8']==sources['repair-review-b-inputs']['utf8'], 'Repair reviewer packs differ')
    for label in ('repair-review-a','repair-review-b'):
        inputs=exact_index([json.loads(line) for line in sources[label+'-inputs']['utf8'].splitlines() if line.strip()],proposed,'repair review inputs')
        for identity,row in inputs.items():
            require(row=={'id':identity,'request':by_id[identity]['request'],'system':by_id[identity]['system'],
                          'reference':repairs[identity]['reference'],'sources':repairs[identity]['sources']}, 'Changed repaired review target/prompt')
        report=parse(label+'-report')
        require(report.get('status')=='completed_provisional_target_review' and report.get('records')==len(proposed)
                and report.get('inputs_sha256')==sources[label+'-inputs']['sha256']
                and report.get('annotations_sha256')==sources[label+'-annotations']['sha256']
                and report.get('repair_proposals_sha256')==sources['repairs-annotations']['sha256']
                and report.get('source_train_sha256')==sha_bytes(bundle['blobs']['train.jsonl']), 'Repair review report binding mismatch')
        review_sets.append(validate_review(parse(label+'-annotations'),proposed,label))
    review_a,review_b=review_sets
    accepted = {i for i in proposed if all(r[i]['quality']=='usable' and r[i]['safety']=='benign'
                                         for r in (review_a,review_b))}
    pool, provenance = [], []
    for original in train:
        identity = original['id']; keep = original['language'] != 'ko' or identity in core or identity in accepted
        origin = 'original_unaudited' if original['language']!='ko' else ('original_both_accepted' if identity in core else 'derived_repair')
        record = {'id':identity, 'retained':keep, 'origin':origin,
                  'original_reference_sha256':sha_bytes(original['reference'].encode())}
        if identity in repairs:
            record['repair'] = repairs[identity]
            record['repair_reviews'] = [review_a.get(identity), review_b.get(identity)]
        if keep:
            row = copy.deepcopy(original)
            if tokenizer is not None and identity not in accepted:
                require(len(tokenizer(row['reference'],add_special_tokens=False)['input_ids'])<=192, f'{identity}: raw target exceeds192 tokens')
            if identity in accepted:
                row['reference'] = repairs[identity]['reference']
                if tokenizer is not None:
                    messages = [{'role':'system','content':row['system']},{'role':'user','content':row['request']}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
                    require(prompt == original['input_ids'][:original['prompt_tokens']], f'{identity}: changed prompt prefix')
                    full = tokenizer.apply_chat_template(messages+[{'role':'assistant','content':row['reference']}],
                                                          tokenize=True, add_generation_prompt=False, return_dict=False)
                    require(full[:len(prompt)] == prompt, f'{identity}: changed prompt tokens')
                    require(tokenizer.decode(full[len(prompt):-1],skip_special_tokens=False)==row['reference'], f'{identity}: target round-trip mismatch')
                    target=tokenizer(row['reference'],add_special_tokens=False)['input_ids']
                    require(len(target)<=192, f'{identity}: target exceeds192 tokens; truncation forbidden')
                    row.update(reference_tokens=len(target),input_ids=full,attention_mask=[1]*len(full),labels=[-100]*len(prompt)+full[len(prompt):],
                               sequence_tokens=len(full),supervised_tokens=len(full)-len(prompt))
            if tokenizer is not None or identity not in accepted:
                validate_pool_row(row,original,bundle)
            record['reference_sha256'] = sha_bytes(row['reference'].encode())
            if tokenizer is not None or identity not in accepted:
                record['tokenized_row_sha256'] = sha_bytes(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False).encode())
            pool.append(row)
        provenance.append(record)
    return pool, provenance, accepted


def validate_pool_row(row, original, bundle):
    for key in ('id','source_id','split','language','request','system','category','prompt_tokens'):
        require(row.get(key) == original.get(key), f'{original["id"]}: changed original {key}')
    require(row['input_ids'][:row['prompt_tokens']] == original['input_ids'][:original['prompt_tokens']], 'Changed prompt prefix')
    v1.validate_token_row(row,max_tokens=1024,eos_token_id=bundle['manifest']['tokenizer']['eos_token_id'])
    if 'reference_tokens' in row:
        require(type(row['reference_tokens']) is int and 0 < row['reference_tokens'] <= 192, f'{row["id"]}: target exceeds192 tokens; truncation forbidden')


def summary(pool, schedules):
    by_id = {r['id']:r for r in pool}
    return {'pool_counts':dict(Counter(r['language'] for r in pool)), 'truncated_examples':0,
            'arms':{arm:{'updates':len(order),'language_exposures':dict(Counter(by_id[i]['language'] for i in order)),
                          'unique_rows':len(set(order)), 'repeated_exposures':len(order)-len(set(order)),
                          'unique_rows_per_language':{lang:len({i for i in order if by_id[i]['language']==lang}) for lang in ('en','ko','zh')},
                          'supervised_tokens_per_language':{lang:sum(by_id[i]['supervised_tokens'] for i in order if by_id[i]['language']==lang) for lang in ('en','ko','zh')},
                          'source_draw_counts':dict(sorted(Counter(order).items()))} for arm,order in schedules.items()}}


def prepare_study(*, prepared, audit_a, audit_b, repairs, repair_review_a, repair_review_b, output, tokenizer=None):
    bundle = v1.load_prepared(prepared)
    output = Path(output); require(not output.exists(), 'Study output already exists')
    sources = {}
    for label,folder in (('audit-a',audit_a),('audit-b',audit_b)):
        for name in ('annotations','report'):
            sources[label+'-'+name] = read_source(Path(folder)/(name+'.json'))
    require(Path(repair_review_a).resolve()!=Path(repair_review_b).resolve(), 'Two distinct repair review directories required')
    for label,folder in (('repairs',repairs),('repair-review-a',repair_review_a),('repair-review-b',repair_review_b)):
        for name,extension in (('inputs','.jsonl'),('annotations','.json'),('report','.json')):
            sources[label+'-'+name]=read_source(Path(folder)/(name+extension))
    if tokenizer is None:
        tokenizer, _, _ = v1.verify_retokenized(bundle)
    require(tokenizer.eos_token_id == bundle['manifest']['tokenizer']['eos_token_id'], 'Tokenizer EOS mismatch')
    pool, records, _ = derive_pool(bundle,sources,tokenizer)
    schedules, alignment_order = build_schedules(pool,bundle['alignment'])
    spec = {'schema_version':1, 'training_method':'safegrad', 'run_kind':'fixed_schedule', 'steps':1000,
            'seed':42,'alignment_seed':43,'quotas':QUOTAS,'schedules':schedules,'alignment_order':alignment_order,
            'original_config':bundle['config'],'original_config_sha256':sha_bytes(bundle['blobs']['config.json']),
            'prepared_manifest_sha256':bundle['manifest_sha256']}
    report = {'status':'passed_study_preparation',**summary(pool,schedules),
              'scope':'Matched update exposures on a common partially audited pool; repeats are not unique examples.'}
    output.mkdir(parents=True,exist_ok=False)
    (output/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in pool),encoding='utf-8')
    write_json(output/'training-config.json',spec)
    write_json(output/'annotations.json',{'sources':sources,'records':records})
    write_json(output/'report.json',report)
    manifest = {'status':'passed_study_preparation','schema_version':1,'prepared_manifest_sha256':bundle['manifest_sha256'],
                'files':{name:sha_bytes((output/name).read_bytes()) for name in sorted(FILES)}}
    write_json(output/'manifest.json',manifest)
    return load_study(output,prepared)


def load_study(directory, prepared):
    directory = Path(directory); bundle = v1.load_prepared(prepared)
    manifest_bytes = (directory/'manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
    require(manifest.get('status')=='passed_study_preparation' and manifest.get('schema_version')==1, 'Invalid study manifest')
    require(manifest.get('prepared_manifest_sha256')==bundle['manifest_sha256'], 'Original prepared manifest mismatch')
    require(set(manifest.get('files',{}))==FILES,'Study file coverage mismatch')
    blobs={}
    for name in sorted(FILES):
        path=directory/name; require(path.is_file() and not path.is_symlink(), 'Missing/symlinked study file')
        blobs[name]=path.read_bytes(); require(sha_bytes(blobs[name])==manifest['files'][name],f'Study hash mismatch: {name}')
    spec=json.loads(blobs['training-config.json']); annotations=json.loads(blobs['annotations.json']);report=json.loads(blobs['report.json'])
    sources=annotations['sources']
    require(set(sources)=={'audit-a-annotations','audit-a-report','audit-b-annotations','audit-b-report'}|{label+'-'+name for label in ('repairs','repair-review-a','repair-review-b') for name in ('inputs','annotations','report')}, 'Missing provenance sources')
    for label,source in sources.items():
        require(isinstance(source.get('utf8'),str) and sha_bytes(source['utf8'].encode())==source.get('sha256'),f'Provenance source hash mismatch: {label}')
        require(not Path(source.get('path','')).is_absolute(), 'Provenance paths must be project relative')
    expected_pool,records,accepted=derive_pool(bundle,sources)
    pool=[json.loads(line) for line in blobs['inputs.jsonl'].splitlines() if line.strip()]
    require(len(pool)==len(expected_pool) and [r['id'] for r in pool]==[r['id'] for r in expected_pool], 'Wrong pool source IDs')
    originals={r['id']:r for r in bundle['train']}
    for row,expected in zip(pool,expected_pool):
        validate_pool_row(row,originals[row['id']],bundle)
        if row['id'] not in accepted:
            require(row==expected,'Original reference/token bytes changed')
        else:
            require(row['reference']==expected['reference'],'Repaired reference changed')
        record=next(r for r in records if r['id']==row['id'])
        record['reference_sha256']=sha_bytes(row['reference'].encode())
        record['tokenized_row_sha256']=sha_bytes(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False).encode())
    require(records==annotations.get('records'),'Retention/repair annotations mismatch')
    schedules,alignment_order=build_schedules(pool,bundle['alignment'])
    expected_spec={'schema_version':1,'training_method':'safegrad','run_kind':'fixed_schedule','steps':1000,
                   'seed':42,'alignment_seed':43,'quotas':QUOTAS,'schedules':schedules,'alignment_order':alignment_order,
                   'original_config':bundle['config'],'original_config_sha256':sha_bytes(bundle['blobs']['config.json']),
                   'prepared_manifest_sha256':bundle['manifest_sha256']}
    require(spec==expected_spec,'Frozen study configuration/schedule mismatch')
    require(report.get('status')=='passed_study_preparation' and all(report.get(k)==v for k,v in summary(pool,schedules).items()), 'Study count/report mismatch')
    return dict(bundle=bundle,pool=pool,pool_by_id={r['id']:r for r in pool},schedules=schedules,
                alignment_order=alignment_order,spec=spec,manifest=manifest,manifest_sha256=sha_bytes(manifest_bytes))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','audit-a','audit-b','repairs','repair-review-a','repair-review-b','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    prepare_study(**vars(args))
    return 0


if __name__=='__main__':raise SystemExit(main())
