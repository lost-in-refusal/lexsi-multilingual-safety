"""CPU-only bounded triage; fake generation, real parser/runtime and journals."""
import copy
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments.tiny_aya import quality, guard, runtime, train
try:
    from experiments.tiny_aya import curate
except ImportError:
    curate = None


def judgment(label='usable', equivalent='yes'):
    value = {lang: dict(quality=label, reason='Visible evidence.') for lang in ('en','ko','zh')}
    return dict(value, equivalent_tasks=equivalent, equivalence_reason='Same task.')


class JudgeTokenizer:
    chat_template = 'fixture official Qwen template'
    def __len__(self): return 151669
    def get_chat_template(self): return self.chat_template
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_dict=False,
                              truncation=False, enable_thinking=False)
        text = train._json(messages)
        return [100] * 4097 if 'INPUT_CAP' in text else [100] + list(hashlib.sha256(text.encode()).digest())
    def encode(self, text, **kwargs): return list(text.encode())
    def decode(self, ids, **kwargs): return bytes(t for t in ids if t < 256).decode()


class TinyTokenizer:
    chat_template = 'fixture official Tiny template'
    eos_token_id = 501
    def __len__(self): return 1024
    def get_chat_template(self): return self.chat_template
    def convert_tokens_to_ids(self, token): return 500 if token == '<|END_RESPONSE|>' else None
    def apply_chat_template(self, messages, **kwargs):
        prompt = [400] + list(messages[0]['content'].encode()) + [401]
        return prompt if len(messages) == 1 else prompt + list(messages[1]['content'].encode()) + [500,501]


class CurateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(curate, 'Task9 triage module is missing')
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root/'sources-v1'; self.prepared.mkdir()
        self.pool = self.root/'pool'; self.pool.mkdir()
        self.review = self.root/'review'; self.review.mkdir()
        self.calibration = self.root/'calibration'
        self.output = self.root/'scores'
        self.ready = self.root/'private'/'readiness.json'; self.ready.parent.mkdir()
        self.originals = [dict(id=f'f{i}:{lang}', family_id=f'f{i}', lang=lang,
            instruction=f'Greet {i}.', response=f'Hello {lang}.', source='source', source_revision='pin',
            provenance=dict(original_instruction=f'Greet {i}.', instruction_changed=False, steps=[]))
            for i in range(2000) for lang in ('en','ko','zh')]
        for directory, rows in ((self.prepared,self.originals[:3000]), (self.pool,self.originals)):
            self.write_pool(directory, rows)
        preview_hashes = {}
        for lang in ('ko','zh'):
            rows=[]; annotations=[]
            for i in range(30):
                row = self.originals[i*3+(1 if lang=='ko' else 2)]
                rows.append({k:v for k,v in row.items() if k!='provenance'} |
                    dict(english_instruction=self.originals[i*3]['instruction'], english_response=self.originals[i*3]['response']))
                annotations.append(dict(id=row['id'], family_id=row['family_id'], lang=lang,
                    quality='usable', reason='Provisional.', material_bad=False))
            self.write_rows(self.review/f'{lang}.jsonl',rows)
            digest=train._hash_file(self.review/f'{lang}.jsonl'); preview_hashes[f'{lang}.jsonl']=digest
            train._write(self.review/f'annotations-{lang}.json',dict(status='provisional_automated_review_not_human_validation',
                source_file=f'{lang}.jsonl',source_sha256=digest,human_review='pending',annotations=annotations))
            train._write(self.review/f'report-{lang}.json',dict(status='provisional_automated_review_not_human_validation',
                source_sha256=digest,human_review='pending',reviewed=30,quality_counts=dict(usable=30),
                material_bad_count=0,material_bad_ids=[]))
        train._write(self.review/'report.json',dict(status='provisional_training_only_preview',human_review='pending',files=preview_hashes))
        self.tokenizer=JudgeTokenizer(); self.tiny=TinyTokenizer(); self.raw=None; self.text=json.dumps(judgment(),separators=(',',':'))
        self.generated=[]
        stack=ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(quality.prepare,'verify_bundle',side_effect=lambda p:quality._read(Path(p)/'manifest.json')))
        stack.enter_context(patch.object(guard,'verify_bundle',side_effect=lambda p:quality._read(Path(p)/'manifest.json')))
        stack.enter_context(patch.object(quality,'_tokenizer',return_value=(self.tokenizer,{'tokenizer.json':'cached'})))
        stack.enter_context(patch.object(quality,'_checkpoint',return_value=(self.root,{'weights.safetensors':'cached'})))
        stack.enter_context(patch.object(quality,'_device',return_value='cpu'))
        stack.enter_context(patch.object(quality,'_new_model',return_value=object()))
        stack.enter_context(patch.object(runtime,'generate_batch',side_effect=self.generate))
        stack.enter_context(patch.object(curate,'EVIDENCE_ROOT',self.root))
        stack.enter_context(patch.object(curate,'_tiny_tokenizer',return_value=(self.tiny,{'tokenizer.json':'tiny-cached'})))
        stack.enter_context(patch.object(curate,'INITIAL_CANDIDATES_SHA256',train._hash_file(self.pool/'candidates.jsonl')))
        quality.calibrate(self.prepared,self.review,self.calibration,rubric_version=2,judge_profile='qwen3-14b')
        self.generated.clear()
        self.write_readiness()

    def write_rows(self,path,rows): path.write_text(''.join(train._json(r)+'\n' for r in rows))
    def write_pool(self,path,rows):
        self.write_rows(path/'candidates.jsonl',rows)
        for name in ('config.json','attacks.jsonl','alignment.jsonl','translation_review.jsonl'):
            (path/name).write_text('{}\n')
        train._write(path/'manifest.json',dict(candidate_families=len(rows)//3,source_files={'fixture':'pins'},curation={'fixture':'curator'},
            files={name:dict(sha256=train._hash_file(path/name),rows=len(rows) if name=='candidates.jsonl' else 1)
                   for name in ('candidates.jsonl','attacks.jsonl','alignment.jsonl','translation_review.jsonl')}))
    def generate(self,model,tokenizer,input_ids,*,generation_config,device):
        self.generated.extend(copy.deepcopy(input_ids))
        raw=self.raw if self.raw is not None else tokenizer.encode(self.text)+[151645]
        return [runtime.completion_evidence(tokenizer,ids,raw,generation_config=generation_config,
            padded_width=len(ids),elapsed_batch_seconds=.01) for ids in input_ids]
    def ref(self,path): return dict(path=os.path.relpath(path,self.ready.parent),sha256=train._hash_file(path))
    def write_readiness(self):
        audit=self.ready.parent/'source-audit.json'; train._write(audit,dict(status='provisional_automated_semantic_audit_not_human_validation',human_review='pending'))
        policy=self.ready.parent/'policy.json'; train._write(policy,dict(status='frozen_before_qwen_outputs',model=quality.QWEN_MODEL,
            revision=quality.QWEN_REVISION,rubric_version=2,semantic_audit_sha256=train._hash_file(audit),human_validation='pending',
            original_reference_labels_unchanged=True,report_all_original_confusion_counts=True,
            confirmed_material_error_rows=[{'row_id':f'f{i}:ko','family_id':f'f{i}'} for i in range(7)],
            criteria=dict(all_30_completed=True,zero_confirmed_error_rows_labeled_usable=True,
                prior_usable_retention_each_language_at_least=.5,at_least_one_family_accepted=True,root_inspects_matching_rejection_evidence=True)))
        semantic=self.ready.parent/'semantic.json'; train._write(semantic,dict(status='provisional_automated_semantic_assessment_not_human_validation',
            source_sha256={os.path.relpath(p,self.root):train._hash_file(p) for p in [audit,policy]+[self.calibration/n for n in ('inputs.jsonl','families.jsonl','run-config.json','report.json')]},
            source_annotation_sha256={os.path.relpath(self.review/n,self.root):train._hash_file(self.review/n) for n in ('annotations-ko.json','annotations-zh.json')}))
        assessment=self.ready.parent/'root.json'; train._write(assessment,dict(status='not_qualified_for_autonomous_quality_approval',full_training_approved=False,
            human_validation='pending',model=quality.QWEN_MODEL,revision=quality.QWEN_REVISION,rubric_version=2,
            evidence={os.path.relpath(p,self.root):train._hash_file(p) for p in (policy,semantic,self.calibration/'manifest.json')}))
        evidence=dict(calibration=self.ref(self.calibration/'manifest.json'),prepared=self.ref(self.prepared/'manifest.json'),
            review_evidence={n:self.ref(self.review/n) for n in ('report.json','ko.jsonl','zh.jsonl','annotations-ko.json','annotations-zh.json','report-ko.json','report-zh.json')},
            frozen_policy=self.ref(policy),source_semantic_audit=self.ref(audit),qwen_semantic_assessment=self.ref(semantic),root_assessment=self.ref(assessment))
        train._write(self.ready,dict(schema_version=1,status='provisional_triage_ready',allowed_use='candidate_triage_only',
            calibration_qualification='not_established',human_review='pending',judge_profile='qwen3-14b',rubric_version=2,
            model=quality.QWEN_MODEL,revision=quality.QWEN_REVISION,evidence=evidence,
            root_decision=dict(route='unchanged_qwen_triage_then_independent_source_audit',qualification_ruling='not_established',decision_basis='Mixed evidence.'),
            data_acceptance=dict(requires_independent_blinded_train_audit=True,families=30,languages=['en','ko','zh'],fail_if_incorrect_per_language_gt=3,max_repairs=1)))
    def score(self,limit=200,**kwargs): return curate.score(self.pool,self.ready,self.output,limit,**kwargs)
    def load(self): return curate.load_scores(self.output,self.pool)

    def test_readiness_keeps_unestablished_ruling_and_all_30_exclusions(self):
        loaded=curate.load_readiness(self.ready)
        self.assertEqual(loaded['exclusions'],[f'f{i}' for i in range(30)])
        self.assertEqual(loaded['record']['calibration_qualification'],'not_established')
        self.assertEqual(loaded['sha256'],train._hash_file(self.ready))
        self.assertEqual(len(loaded['calibration']['rows']),30)

    def test_readiness_rejects_approval_and_binding_tampering(self):
        original=quality._read(self.ready)
        changes=[('calibration_qualification','qualified'),('human_review','validated'),('allowed_use','data_approved'),('rubric_version',1),('judge_profile','aya-expanse-8b')]
        for field,value in changes:
            with self.subTest(field=field):
                train._write(self.ready,{**original,field:value})
                with self.assertRaises(ValueError):curate.load_readiness(self.ready)
        train._write(self.ready,original)
        policy=self.ready.parent/'policy.json'; policy.write_text('{}\n')
        with self.assertRaises(ValueError):curate.load_readiness(self.ready)

    def test_pilot_is_exact_200_inspected_prefix_and_resume_stops_at_900(self):
        report=self.score()
        self.assertEqual((report['status'],report['families_inspected'],report['triage_selected']),('pilot_complete',200,200))
        loaded=self.load()
        self.assertEqual(loaded['selected_family_ids'],[f'f{i}' for i in range(30,230)])
        self.assertEqual(loaded['rows'][0]['source_index'],30)
        self.assertEqual(loaded['families'][0]['rows'],self.originals[90:93])
        pilot=(self.output/'pilot-report.json').read_bytes()
        first=(self.output/'families.jsonl').read_bytes()
        self.score(None)
        self.assertEqual(len(self.generated),900)
        loaded=self.load()
        self.assertEqual((loaded['report']['status'],len(loaded['selected_family_ids'])),('target_reached',900))
        self.assertTrue((self.output/'families.jsonl').read_bytes().startswith(first))
        self.assertEqual((self.output/'pilot-report.json').read_bytes(),pilot)
        self.assertFalse(loaded['report']['full_training_approved'])
        self.assertEqual(loaded['report']['human_review'],'pending')

    def test_tiny_preflight_and_input_cap_skip_generation_and_count_as_inspected(self):
        rows=copy.deepcopy(self.originals); rows[3000]['response']='x'*511; rows[3003]['instruction']='INPUT_CAP'
        self.write_pool(self.pool,rows)
        with patch.object(curate,'INITIAL_CANDIDATES_SHA256',train._hash_file(self.pool/'candidates.jsonl')):
            self.write_pool(self.prepared,self.originals[:3000]); self.write_readiness()
            self.text=json.dumps(judgment('unclear'))
            self.score(973)
            loaded=self.load()
        self.assertEqual([r['state'] for r in loaded['rows'][-3:]],['tiny_preflight_reject','quality_input_cap','generated'])
        self.assertEqual(len(self.generated),971)
        self.assertEqual(loaded['rows'][-3]['judgment']['labels']['en'],'unknown')

    def test_unknown_and_one_bad_member_never_select_whole_family(self):
        self.text=json.dumps(judgment('unclear'))
        self.score(1)
        self.assertEqual(self.load()['selected_family_ids'],[])
        value=judgment();value['ko']['quality']='incorrect'; self.text=json.dumps(value)
        self.score(2)
        self.assertEqual(self.load()['selected_family_ids'],[])
        self.text='```json {} ```';self.score(3)
        self.assertEqual(self.load()['rows'][2]['judgment']['state'],'unknown')

    def test_readiness_rejects_path_escape_missing_review_and_rehashed_crosslinks(self):
        original=quality._read(self.ready)
        for mutation in ('absolute','escape','review','crosslink','training','timestamp','policy'):
            with self.subTest(mutation=mutation):
                self.write_readiness(); value=quality._read(self.ready)
                if mutation=='absolute':value['evidence']['prepared']['path']=str(self.prepared/'manifest.json')
                elif mutation=='escape':value['evidence']['prepared']['path']='../../../outside.json'
                elif mutation=='review':del value['evidence']['review_evidence']['ko.jsonl']
                elif mutation=='timestamp':value['created_at_utc']='2026-11-04T00:00:00'
                else:
                    key={'crosslink':'qwen_semantic_assessment','training':'root_assessment','policy':'frozen_policy'}[mutation]
                    path=self.ready.parent/value['evidence'][key]['path']; artifact=quality._read(path)
                    if mutation=='crosslink':artifact['source_sha256']['calibration/inputs.jsonl']='0'*64
                    elif mutation=='training':artifact['full_training_approved']=True
                    else:artifact['semantic_audit_sha256']='0'*64
                    train._write(path,artifact); value['evidence'][key]=self.ref(path)
                train._write(self.ready,value)
                with self.assertRaises(ValueError):curate.load_readiness(self.ready)
        self.write_readiness()
        (self.ready.parent/'link.json').symlink_to('/etc/hosts')
        value=quality._read(self.ready);value['evidence']['prepared']=self.ref(self.ready.parent/'link.json')
        train._write(self.ready,value)
        with self.assertRaises(ValueError):curate.load_readiness(self.ready)

    def test_readiness_rejects_incomplete_and_legacy_calibration(self):
        shutil.rmtree(self.calibration)
        quality.calibrate(self.prepared,self.review,self.calibration,1,rubric_version=2,judge_profile='qwen3-14b')
        self.write_readiness()
        with self.assertRaises(ValueError):curate.load_readiness(self.ready)
        shutil.rmtree(self.calibration)
        quality.calibrate(self.prepared,self.review,self.calibration,rubric_version=1,judge_profile='qwen3-14b')
        self.write_readiness()
        with self.assertRaises(ValueError):curate.load_readiness(self.ready)

    def test_raw_stop_reconstruction_handles_eos_pad_and_eos_at_cap(self):
        self.raw=self.tokenizer.encode(self.text)+[151643,151643]
        self.score(1)
        row=self.load()['rows'][0]
        self.assertTrue(row['triage_selected'])
        self.assertEqual(row['evidence']['post_eos_pad_count'],1)
        self.raw=self.tokenizer.encode(self.text)
        self.raw += [32]*(383-len(self.raw))+[151645]
        self.score(2)
        row=self.load()['rows'][1]
        self.assertTrue(row['triage_selected'])
        self.assertEqual(row['evidence']['completion_tokens'],384)
        self.raw=self.tokenizer.encode(self.text)+[32]*(384-len(self.tokenizer.encode(self.text)))
        self.score(3)
        row=self.load()['rows'][2]
        self.assertFalse(row['triage_selected'])
        self.assertEqual(row['judgment']['unknown_reason'],'incomplete_completion')

    def test_corrupt_or_short_raw_generation_fails_and_keeps_completed_prefix(self):
        self.score(1)
        self.raw=self.tokenizer.encode(self.text) # Short, with neither EOS nor cap.
        with self.assertRaises(ValueError):self.score(2)
        loaded=self.load()
        self.assertEqual(loaded['report']['status'],'failed')
        self.assertEqual(len(loaded['rows']),1)
        self.raw=None;self.score(2)
        self.assertEqual(self.load()['report']['families_inspected'],2)

    def test_rehashed_runtime_judgment_and_report_tampering_are_reconstructed(self):
        self.score(2)
        ledger=(self.output/'families.jsonl').read_bytes()
        report=(self.output/'report.json').read_bytes()
        manifest=(self.output/'manifest.json').read_bytes()
        for mutation in ('metadata','decision','report','state','input','duplicate','reorder','torn','extra'):
            with self.subTest(mutation=mutation):
                (self.output/'families.jsonl').write_bytes(ledger)
                (self.output/'report.json').write_bytes(report)
                (self.output/'manifest.json').write_bytes(manifest)
                rows=quality._rows(self.output/'families.jsonl')
                if mutation=='metadata':rows[0]['evidence']['response']='{}'
                elif mutation=='decision':rows[0]['triage_selected']=False
                elif mutation=='state':rows[0]['state']='quality_input_cap';rows[0]['evidence']=None
                elif mutation=='input':rows[0]['input_sha256']='0'*64
                elif mutation=='duplicate':rows[1]=rows[0]
                elif mutation=='reorder':rows.reverse()
                elif mutation=='extra':(self.output/'extra.json').write_text('{}')
                if mutation=='torn':(self.output/'families.jsonl').write_bytes(ledger[:-1])
                else:self.write_rows(self.output/'families.jsonl',rows)
                if mutation=='report':
                    value=quality._read(self.output/'report.json');value['triage_selected']=0;train._write(self.output/'report.json',value)
                # Hash-only validation must not be enough to accept an edited decision.
                value=quality._read(self.output/'manifest.json')
                for name in ('families.jsonl','report.json'):
                    if mutation!='torn':value['files'][name]=quality._file_entry(self.output/name)
                train._write(self.output/'manifest.json',value)
                with self.assertRaises(ValueError):self.load()
                (self.output/'extra.json').unlink(missing_ok=True)

    def test_writer_lock_config_mutation_and_source_mutation_refuse_resume(self):
        self.score(1)
        lock=self.output/'.writer.lock';lock.write_text('{}')
        with self.assertRaises((ValueError,RuntimeError)):self.load()
        with self.assertRaises(RuntimeError):self.score(2)
        self.assertTrue(lock.exists());lock.unlink()
        config=(self.output/'run-config.json').read_bytes()
        value=quality._read(self.output/'run-config.json');value['settings']['batch_size']=2;train._write(self.output/'run-config.json',value)
        with self.assertRaises(ValueError):self.score(2)
        (self.output/'run-config.json').write_bytes(config)
        path=self.output/'sources'/'curate.py';path.write_text('mutated')
        with self.assertRaises(ValueError):self.load()
        with self.assertRaises(ValueError):self.score(2)

    def test_reader_accepts_archived_sources_but_writer_requires_current_executing_source(self):
        self.score(1)
        other=self.root/'later-train.py';other.write_text('later integration source')
        paths=curate._source_paths();paths['train.py']=other
        with patch.object(curate,'_source_paths',return_value=paths):
            self.assertEqual(self.load()['report']['families_inspected'],1)
            with self.assertRaises(ValueError):self.score(2)

    def test_source_prefix_and_auxiliary_changes_refuse_before_generation(self):
        rows=copy.deepcopy(self.originals);rows[1]['response']='Changed original response'
        self.write_pool(self.pool,rows)
        with patch.object(curate,'INITIAL_CANDIDATES_SHA256',train._hash_file(self.pool/'candidates.jsonl')):
            with self.assertRaises(ValueError):self.score(1)
        self.write_pool(self.pool,self.originals);(self.pool/'alignment.jsonl').write_text('changed\n')
        with self.assertRaises(ValueError):self.score(1)
        self.assertEqual(self.generated,[])

    def test_extension_inherits_original_configs_without_prefix_inference(self):
        self.text=json.dumps(judgment('unclear'),separators=(',',':'))
        report=self.score(None)
        self.assertEqual((report['status'],report['families_inspected']),('pool_exhausted_insufficient',1970))
        raw=(self.output/'families.jsonl').read_bytes(); original_config=(self.output/'run-config.json').read_bytes()
        extended=self.root/'extended';extended.mkdir()
        tail=[dict(id=f'f{i}:{lang}',family_id=f'f{i}',lang=lang,instruction=f'Greet {i}.',response=f'Hello {lang}.',
            source='source',source_revision='pin',provenance=dict(original_instruction=f'Greet {i}.',instruction_changed=False,steps=[]))
            for i in range(2000,5000) for lang in ('en','ko','zh')]
        self.write_pool(extended,self.originals+tail)
        self.text=json.dumps(judgment(),separators=(',',':'))
        destination=self.root/'extended-scores'
        report=curate.score(extended,self.ready,destination,1972,extension_of=self.output)
        self.assertEqual((report['families_inspected'],report['triage_selected']),(1972,2))
        loaded=curate.load_scores(destination,extended)
        self.assertEqual(loaded['selected_family_ids'],['f2000','f2001'])
        self.assertEqual(len(self.generated),1972)
        self.assertEqual(loaded['rows'][1969]['config_sha256'],train._hash_file(self.output/'run-config.json'))
        self.assertEqual(loaded['rows'][1970]['config_sha256'],train._hash_file(destination/'run-config.json'))
        self.assertEqual(len(loaded['families']),4970)
        self.assertEqual((destination/'pilot-report.json').read_bytes(),(self.output/'pilot-report.json').read_bytes())
        curate.score(extended,self.ready,destination,None,extension_of=self.output)
        self.assertEqual(curate.load_scores(destination,extended)['report']['status'],'target_reached')
        self.assertEqual(len(self.generated),2870)
        self.assertEqual((self.output/'families.jsonl').read_bytes(),raw)
        self.assertEqual((self.output/'run-config.json').read_bytes(),original_config)
        with self.assertRaises(ValueError):curate.score(extended,self.ready,self.root/'second-extension',extension_of=destination)

    def test_extension_requires_closed_exhausted_parent_and_exact_prefix(self):
        self.score(1)
        with self.assertRaises(ValueError):curate.score(self.pool,self.ready,self.root/'extension',extension_of=self.output)

    def test_no_records_after_terminal_target_and_pilot_prefix_is_immutable(self):
        self.score(None)
        rows=quality._rows(self.output/'families.jsonl');rows.append(copy.deepcopy(rows[-1]))
        self.write_rows(self.output/'families.jsonl',rows)
        with self.assertRaises(ValueError):self.load()

    def test_cli_readiness_score_and_verify(self):
        with patch('builtins.print'):
            self.assertEqual(curate.main(['readiness','--triage-readiness',str(self.ready)])['allowed_use'],'candidate_triage_only')
            report=curate.main(['score','--prepared',str(self.pool),'--triage-readiness',str(self.ready),
                '--output',str(self.output),'--max-families','1'])
            self.assertEqual(report['status'],'incomplete')
            verified=curate.main(['verify','--prepared',str(self.pool),'--output',str(self.output)])
            self.assertEqual(verified['families_inspected'],1)

    def test_rehashed_frozen_input_tokens_and_pilot_are_not_trusted(self):
        self.score()
        frozen=(self.output/'inputs.jsonl').read_bytes()
        values=quality._rows(self.output/'inputs.jsonl'); values[0]['tiny_preflight']['rows'][0]['input_ids'][0]=0
        self.write_rows(self.output/'inputs.jsonl',values)
        manifest=quality._read(self.output/'manifest.json'); manifest['files']['inputs.jsonl']=quality._file_entry(self.output/'inputs.jsonl')
        train._write(self.output/'manifest.json',manifest)
        with self.assertRaises(ValueError):self.load()
        (self.output/'inputs.jsonl').write_bytes(frozen)
        pilot=quality._read(self.output/'pilot-report.json');pilot['triage_selected']=199;train._write(self.output/'pilot-report.json',pilot)
        manifest['files']['inputs.jsonl']=quality._file_entry(self.output/'inputs.jsonl')
        manifest['files']['pilot-report.json']=quality._file_entry(self.output/'pilot-report.json');train._write(self.output/'manifest.json',manifest)
        with self.assertRaises(ValueError):self.load()
        with self.assertRaises(ValueError):self.score(201)

    def test_lock_is_held_through_manifest_verification_and_generation(self):
        self.score(1)
        generate=self.generate
        def checked(*args,**kwargs):
            self.assertTrue((self.output/'.writer.lock').exists())
            return generate(*args,**kwargs)
        with patch.object(runtime,'generate_batch',side_effect=checked):self.score(2)
        self.assertFalse((self.output/'.writer.lock').exists())

    def test_generation_failure_retains_successful_family_then_resumes_without_reinference(self):
        generate=self.generate; calls=[]
        def fail_second(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise FloatingPointError('Nonfinite fake logits')
            return generate(*args,**kwargs)
        with patch.object(runtime,'generate_batch',side_effect=fail_second):
            with self.assertRaises(FloatingPointError):self.score(3)
        loaded=self.load()
        self.assertEqual((loaded['report']['status'],len(loaded['rows'])),('failed',1))
        prefix=(self.output/'families.jsonl').read_bytes()
        self.score(3)
        self.assertEqual(len(self.generated),3)
        self.assertTrue((self.output/'families.jsonl').read_bytes().startswith(prefix))

    def test_budget_validation_and_shrinking_prefix_refuse(self):
        for limit in (True,0,-1,1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):self.score(limit)
        self.score(2)
        with self.assertRaises(ValueError):self.score(1)

    def test_preflight_only_prefix_never_loads_model_and_contains_no_outputs(self):
        # Rejects beyond the anchored first1,000 allow a real unchanged source prefix.
        rows=copy.deepcopy(self.originals)
        for row in rows[3000:]:row['response']='x'*511
        self.write_pool(self.pool,rows)
        self.text=json.dumps(judgment('unclear'),separators=(',',':'))
        with patch.object(curate,'INITIAL_CANDIDATES_SHA256',train._hash_file(self.pool/'candidates.jsonl')):
            self.score(970)
            with patch.object(quality,'_new_model',side_effect=AssertionError('Preflight must skip the model')):
                self.score(973)
            loaded=self.load()
        self.assertEqual(len(self.generated),970)
        for row in loaded['rows'][-3:]:
            self.assertEqual(row['state'],'tiny_preflight_reject')
            self.assertIsNone(row['evidence'])
            self.assertFalse(row['triage_selected'])

    def test_relative_evidence_copy_can_move_without_absolute_private_paths(self):
        self.score(1)
        copied=self.root/'copy';shutil.copytree(self.root,copied,ignore=shutil.ignore_patterns('copy'))
        with patch.object(curate,'EVIDENCE_ROOT',copied):
            loaded=curate.load_scores(copied/'scores',copied/'pool')
        self.assertEqual(loaded['selected_family_ids'],['f30'])
        config=quality._read(copied/'scores'/'run-config.json')
        self.assertFalse(Path(config['bindings']['readiness']['path']).is_absolute())
        self.assertFalse(Path(config['prepared_path']).is_absolute())
        self.assertEqual((copied/'scores'/'run-config.json').read_bytes(),(self.output/'run-config.json').read_bytes())

    def test_parser_task_equivalence_and_uncertainty_remain_distinct(self):
        for i,(label,equivalence) in enumerate((('usable','no'),('usable','unclear'),('ambiguous','yes'),('incorrect','yes')),1):
            self.text=json.dumps(judgment(label,equivalence),separators=(',',':')); self.score(i)
        rows=self.load()['rows']
        self.assertEqual([r['judgment']['labels']['en'] for r in rows],['usable','usable','ambiguous','incorrect'])
        self.assertEqual([r['judgment']['equivalent_tasks'] for r in rows],['no','unclear','yes','yes'])
        self.assertEqual(self.load()['selected_family_ids'],[])

    def test_archived_source_symlink_is_not_a_frozen_snapshot(self):
        self.score(1)
        path=self.output/'sources'/'train.py';path.unlink();path.symlink_to(Path(train.__file__).resolve())
        with self.assertRaises(ValueError):self.load()

    def test_extension_rejects_changed_prefix_then_honestly_exhausts_absolute_5000_ceiling(self):
        self.text=json.dumps(judgment('unclear'),separators=(',',':'));self.score(None)
        extended=self.root/'extended';extended.mkdir()
        tail=[{**row,'id':f'f{i}:{row["lang"]}','family_id':f'f{i}'}
            for i in range(2000,5000) for row in self.originals[:3]]
        rows=self.originals+tail
        changed=copy.deepcopy(rows);changed[3001]['response']='Altered inherited response'
        self.write_pool(extended,changed)
        with self.assertRaises(ValueError):curate.score(extended,self.ready,self.root/'extension',extension_of=self.output)
        self.assertEqual(len(self.generated),1970)
        self.write_pool(extended,rows)
        with self.assertRaises(ValueError):
            curate.score(extended,self.ready,self.output/'nested-extension',1971,extension_of=self.output)
        destination=self.root/'extension'
        report=curate.score(extended,self.ready,destination,None,extension_of=self.output)
        self.assertEqual((report['status'],report['families_inspected'],report['triage_selected']),('pool_exhausted_insufficient',4970,0))
        self.assertEqual(len(self.generated),4970)
        self.assertEqual(curate.load_scores(destination,extended)['report']['last_source_index'],4999)
        with self.assertRaises(ValueError):curate.score(extended,self.ready,self.root/'second',extension_of=destination)
