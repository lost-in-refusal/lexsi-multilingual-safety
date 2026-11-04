"""CPU-only quality calibration: strict labels and original-family evidence."""
import copy
import json
import unittest

try:
    from experiments.tiny_aya import quality
except ImportError:
    quality = None


def judgment(**changes):
    value = {lang: {'quality': 'usable', 'reason': 'Visible task is satisfied.'}
             for lang in ('en', 'ko', 'zh')}
    value.update(equivalent_tasks='yes', equivalence_reason='Same visible task.')
    value.update(changes)
    return value


class ParserTests(unittest.TestCase):
    def parse(self, text, finish='eos'):
        self.assertIsNotNone(quality, 'The bounded quality calibration module is missing')
        return quality.parse_judgment(text, finish_reason=finish)

    def test_family_acceptance_requires_all_languages_and_equivalence(self):
        parsed = self.parse(json.dumps(judgment()))
        self.assertTrue(parsed['family_pass'])
        for lang in ('en', 'ko', 'zh'):
            for label in ('incorrect', 'ambiguous', 'unclear'):
                value = judgment(**{lang: {'quality': label, 'reason': 'Specific visible issue.'}})
                parsed = self.parse(json.dumps(value))
                self.assertFalse(parsed['family_pass'])
                self.assertEqual(parsed['labels'][lang], label)
        for equivalent in ('no', 'unclear'):
            self.assertFalse(self.parse(json.dumps(judgment(equivalent_tasks=equivalent)))['family_pass'])

    def test_duplicate_keys_wrong_schema_and_invalid_reasons_are_unknown(self):
        valid = json.dumps(judgment())
        invalid = [valid.replace('"equivalent_tasks": "yes"', '"equivalent_tasks": "no", "equivalent_tasks": "yes"'),
                   valid.replace('"quality": "usable"', '"quality": "incorrect", "quality": "usable"', 1),
                   '```json\n' + valid + '\n```', valid + ' trailing', '[]', 'null']
        for changes in ({'extra': 1}, {'en': {'quality': True, 'reason': 'issue'}},
                        {'en': {'quality': 'unknown', 'reason': 'issue'}},
                        {'en': {'quality': 'usable', 'reason': ''}},
                        {'en': {'quality': 'usable', 'reason': 'a' * 241}},
                        {'equivalence_reason': False}, {'equivalent_tasks': True}):
            invalid.append(json.dumps(judgment(**changes)))
        value = judgment();del value['zh'];invalid.append(json.dumps(value))
        for text in invalid:
            with self.subTest(text=text[:90]):
                result = self.parse(text)
                self.assertFalse(result['family_pass'])
                self.assertEqual(result['labels'], dict(en='unknown', ko='unknown', zh='unknown'))
                self.assertEqual(result['state'], 'unknown')

    def test_non_eos_never_passes_even_with_valid_json(self):
        result = self.parse(json.dumps(judgment()), finish='length')
        self.assertEqual(result['state'], 'unknown')
        self.assertFalse(result['family_pass'])

    def test_prompt_uses_only_visible_pairs_never_prior_labels_or_reason(self):
        self.assertIsNotNone(quality, 'The bounded quality calibration module is missing')
        rows = [dict(id=f'f:{lang}', family_id='f', lang=lang, instruction='Classify this.',
                     response=f'Answer in {lang}', quality='PRIOR_LABEL_SECRET', reason='PRIOR_REASON_SECRET')
                for lang in ('en', 'ko', 'zh')]
        messages = quality.family_messages(rows)
        rendered = json.dumps(messages)
        self.assertNotIn('PRIOR_LABEL_SECRET', rendered)
        self.assertNotIn('PRIOR_REASON_SECRET', rendered)
        self.assertIn('Answer in ko', rendered)
        self.assertIn('Classify this.', rendered)
        self.assertIn('Neither English nor any other response is a gold answer.', rendered)

    def test_report_separates_unknown_from_known_bad_and_detects_reject_all(self):
        self.assertIsNotNone(quality, 'The bounded quality calibration module is missing')
        references = {'f0': {'ko': 'incorrect', 'zh': 'usable'},
                      'f1': {'ko': 'usable', 'zh': 'incorrect'}}
        unknown = self.parse('{}')
        rows = [dict(family_id=f'f{i}', judgment=unknown, state='generated') for i in range(2)]
        report = quality.comparison_report(rows, references, total=2)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['languages']['ko']['confusion']['incorrect']['unknown'], 1)
        self.assertEqual(report['languages']['ko']['confusion']['usable']['unknown'], 1)
        self.assertEqual(report['known_bad_families_accepted'], 0)
        self.assertEqual(report['unknown_families'], 2)
        self.assertFalse(report['operational_gates']['at_least_one_family_accepted'])
        self.assertFalse(report['operational_gates']['prior_usable_retention_at_least_half_each_language'])
        self.assertEqual(report['suitability'], 'unestablished_pending_root_evidence_inspection')
        self.assertEqual(report['human_review'], 'pending')
        rows[0]['judgment'] = self.parse(json.dumps(judgment()))
        report = quality.comparison_report(rows, references, total=2)
        self.assertEqual(report['languages']['ko']['known_bad_members_labeled_usable'], 1)
        self.assertEqual(report['known_bad_families_accepted'], 1)
        self.assertFalse(report['operational_gates']['zero_known_bad_families_accepted'])

from contextlib import ExitStack
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiments.tiny_aya import runtime, train


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from experiments.tiny_aya.guard import _snapshot
        snapshot, cls.tokenizer_files = _snapshot('CohereLabs/aya-expanse-8b',
            '5062468bf9bc0c6035fd64e06274333ec127d980', False)
        cls.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)

    def setUp(self):
        self.assertTrue(hasattr(quality, 'calibrate'), 'Bounded calibration entrypoint is missing')
        self.temporary = tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root/'prepared';self.prepared.mkdir()
        self.review = self.root/'review';self.review.mkdir()
        self.output = self.root/'output'
        self.originals = [dict(id=f'f{i}:{lang}', family_id=f'f{i}', lang=lang,
            instruction=f'Give a short greeting {i}.', response=f'Hello {lang} {i}.',
            source='MBZUAI/Bactrian-X', source_revision='3698480a001cb3f62c61c314d4acb181eb31e983',
            provenance={'original_instruction': f'Give a short greeting {i}.'})
            for i in range(32) for lang in ('en','ko','zh')]
        self.write_rows(self.prepared/'candidates.jsonl', self.originals)
        train._write(self.prepared/'manifest.json', {'fixture': 'prepared-verified-at-boundary'})
        index = {r['id']: r for r in self.originals}
        source_hashes = {}
        for lang in ('ko','zh'):
            preview = []
            annotations = []
            for i in range(30):
                source = index[f'f{i}:{lang}'];en = index[f'f{i}:en']
                preview.append({k: v for k,v in source.items() if k != 'provenance'} |
                    dict(english_instruction=en['instruction'], english_response=en['response']))
                label = 'incorrect' if i == (0 if lang == 'ko' else 1) else 'usable'
                annotations.append(dict(id=source['id'], family_id=source['family_id'], lang=lang,
                    quality=label, material_bad=label=='incorrect', reason='PRIVATE_PRIOR_REASON',
                    human_review='pending'))
            path = self.review/f'{lang}.jsonl';self.write_rows(path, preview)
            digest = train._hash_file(path);source_hashes[path.name] = digest
            train._write(self.review/f'annotations-{lang}.json', dict(
                status='provisional_automated_review_not_human_validation', source_file=path.name,
                source_sha256=digest, annotations=annotations, human_review='pending'))
            train._write(self.review/f'report-{lang}.json', dict(
                status='provisional_automated_review_not_human_validation', source_sha256=digest,
                reviewed=30, quality_counts=dict(usable=29, incorrect=1),
                material_bad_count=1, material_bad_ids=[f'f{0 if lang=="ko" else 1}:{lang}'], human_review='pending'))
        train._write(self.review/'report.json', dict(status='provisional_training_only_preview',
            files=source_hashes, human_review='pending'))
        self.snapshot = self.root/'checkpoint';self.snapshot.mkdir()
        train._write(self.snapshot/'config.json', dict(model_type='cohere', num_hidden_layers=32,
            hidden_size=4096, vocab_size=256000))
        self.files = {'config.json': train._hash_file(self.snapshot/'config.json'), 'weights.safetensors': 'fixture-only'}
        self.calls = []
        self.raw_text = json.dumps(judgment())
        self.raw_override = None
        stack = ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(quality.prepare, 'verify_bundle', return_value={'fixture':'verified'}))
        stack.enter_context(patch.object(quality, '_tokenizer', return_value=(self.tokenizer, self.tokenizer_files)))
        stack.enter_context(patch.object(quality, '_checkpoint', return_value=(self.snapshot, self.files)))
        stack.enter_context(patch.object(quality, '_new_model', return_value=object()))
        stack.enter_context(patch.object(quality, '_device', return_value='cpu'))
        stack.enter_context(patch.object(quality.runtime, 'generate_batch', side_effect=self.fake_generate))

    def write_rows(self, path, rows):
        path.write_text(''.join(train._json(r)+'\n' for r in rows), encoding='utf-8')

    def fake_generate(self, model, tokenizer, input_ids, *, generation_config, device):
        self.calls.extend(input_ids)
        raw = self.raw_override if self.raw_override is not None else tokenizer.encode(self.raw_text, add_special_tokens=False)+[generation_config.eos_token_id[0]]
        return [runtime.completion_evidence(tokenizer, ids, raw, generation_config=generation_config,
            padded_width=len(ids), elapsed_batch_seconds=.1) for ids in input_ids]

    def calibrate(self, **kwargs):
        return quality.calibrate(self.prepared, self.review, self.output, **kwargs)

    def load(self):
        return quality.load_calibration(self.output, self.prepared, self.review)

    def rehash(self):
        manifest = quality._read(self.output/'manifest.json')
        for name in manifest['files']:
            manifest['files'][name] = quality._file_entry(self.output/name)
        train._write(self.output/'manifest.json',manifest)

    def test_original_family_order_prefix_resume_and_cpu_verification(self):
        first = self.calibrate(max_families=1)
        self.assertEqual(first['status'], 'incomplete')
        self.assertEqual(first['families_reviewed'], 1)
        prefix = (self.output/'families.jsonl').read_bytes()
        loaded = self.load()
        self.assertEqual(loaded['families'][0]['rows'], self.originals[:3])
        self.assertEqual([r['lang'] for r in loaded['families'][0]['rows']], ['en','ko','zh'])
        prompt = self.tokenizer.decode(self.calls[0],skip_special_tokens=False)
        self.assertNotIn('PRIVATE_PRIOR_REASON', prompt)
        self.assertNotIn('material_bad', prompt)
        self.assertNotIn('provenance', prompt)
        self.calibrate(max_families=2)
        self.assertTrue((self.output/'families.jsonl').read_bytes().startswith(prefix))
        self.assertEqual(len(self.load()['rows']), 2)
        self.calibrate()
        loaded = self.load()
        self.assertEqual(loaded['report']['status'], 'complete')
        self.assertEqual([r['family_id'] for r in loaded['rows']], [f'f{i}' for i in range(30)])
        self.assertEqual(loaded['report']['known_bad_families_accepted'], 2)
        self.assertFalse(loaded['report']['computational_gates_met'])
        self.assertEqual(loaded['config']['generation']['eos_token_id'], [255001])
        self.assertEqual(loaded['config']['generation']['pad_token_id'], 0)
        self.assertEqual(loaded['config']['generation']['max_new_tokens'], 384)
        self.assertNotIn(str(self.root), json.dumps(loaded))

    def test_reference_source_and_annotation_binding_rejects_wrong_rows_and_order(self):
        for target, change in [
            ('ko.jsonl', lambda rows: rows[0].update(response='Rewritten source')),
            ('ko.jsonl', lambda rows: rows.reverse()),
            ('annotations-ko.json', lambda data: data['annotations'][0].update(material_bad=False)),
            ('annotations-zh.json', lambda data: data['annotations'][0].update(id='missing:zh'))]:
            path = self.review/target;before = path.read_bytes()
            value = quality._rows(path) if path.suffix=='.jsonl' else quality._read(path)
            change(value)
            self.write_rows(path, value) if path.suffix=='.jsonl' else train._write(path,value)
            with self.subTest(target=target), self.assertRaises(ValueError):self.calibrate(max_families=1)
            self.assertFalse(self.output.exists())
            path.write_bytes(before)

    def test_changed_executing_source_blocks_resume_but_archive_is_readable(self):
        self.calibrate(max_families=1)
        before = {p.relative_to(self.output):p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        sources = quality._source_paths()
        changed = self.root/'changed.py';changed.write_bytes(sources['quality.py'].read_bytes()+b'\n# changed\n')
        with patch.object(quality,'_source_paths',return_value={**sources,'quality.py':changed}):
            with self.assertRaisesRegex(ValueError,'Current implementation source'):self.calibrate(max_families=2)
            self.assertEqual(len(self.load()['rows']),1)
        self.assertEqual(before,{p.relative_to(self.output):p.read_bytes() for p in self.output.rglob('*') if p.is_file()})

    def test_rehashed_token_text_label_and_context_tampering_are_rejected(self):
        self.calibrate(max_families=1)
        path = self.output/'families.jsonl';original = path.read_bytes()
        changes = [lambda row: row['evidence'].update(response='Forged usable JSON'),
            lambda row: row['evidence']['raw_generated_ids'].__setitem__(0,True),
            lambda row: row['judgment']['labels'].update(ko='incorrect'),
            lambda row: row.update(input_sha256='forged'),
            lambda row: row.update(family_index=1),
            lambda row: row['evidence']['input_ids'].__setitem__(0,0)]
        for change in changes:
            row = quality._rows(path)[0];change(row);self.write_rows(path,[row]);self.rehash()
            with self.subTest(change=change), self.assertRaises(ValueError):self.load()
            with self.assertRaises(ValueError):self.calibrate(max_families=2)
            path.write_bytes(original);self.rehash()
        path.write_bytes(original[:-1])
        with self.assertRaises(ValueError):self.load()
        with self.assertRaises(ValueError):self.calibrate(max_families=2)

    def test_eos_at_384_completes_but_non_eos_cap_is_unknown(self):
        tokens = self.tokenizer.encode(json.dumps(judgment()),add_special_tokens=False)
        spaces = self.tokenizer.encode(' ',add_special_tokens=False)
        self.assertEqual(len(spaces),1)
        self.raw_override = tokens + spaces*(383-len(tokens)) + [255001]
        self.calibrate(max_families=1)
        row = self.load()['rows'][0]
        self.assertTrue(row['judgment']['family_pass'])
        self.assertEqual(row['evidence']['completion_tokens'],384)
        self.raw_override = tokens + spaces*(384-len(tokens))
        self.calibrate(max_families=2)
        loaded = self.load()
        self.assertFalse(loaded['rows'][1]['judgment']['family_pass'])
        self.assertEqual(loaded['report']['output_cap_families'],1)
        self.assertEqual(loaded['report']['unknown_families'],1)

    def test_no_truncation_input_caps_and_invalid_limits_or_overlapping_paths(self):
        for value in (True,0,-1,30,31):
            with self.subTest(value=value), self.assertRaises(ValueError):self.calibrate(max_families=value)
        self.assertFalse(self.output.exists())
        # Actual tokenizer measures the whole rubric and visible source; never shorten it.
        for row in self.originals[:3]:row['response']='Long source answer. '*5000
        self.write_rows(self.prepared/'candidates.jsonl',self.originals)
        for lang in ('ko','zh'):
            preview=quality._rows(self.review/f'{lang}.jsonl')
            preview[0]['response']=self.originals[1 if lang=='ko' else 2]['response']
            preview[0]['english_response']=self.originals[0]['response']
            self.write_rows(self.review/f'{lang}.jsonl',preview)
            digest=train._hash_file(self.review/f'{lang}.jsonl')
            for name in (f'annotations-{lang}.json',f'report-{lang}.json'):
                value=quality._read(self.review/name);value['source_sha256']=digest;train._write(self.review/name,value)
            value=quality._read(self.review/'report.json');value['files'][f'{lang}.jsonl']=digest
            train._write(self.review/'report.json',value)
        report=self.calibrate(max_families=1)
        self.assertEqual(report['input_cap_families'],1)
        self.assertEqual(self.calls,[])
        self.assertFalse(self.load()['rows'][0]['judgment']['family_pass'])

    def test_lock_no_stealing_and_immutable_reference_model_and_config(self):
        self.calibrate(max_families=1)
        lock=self.output/'.writer.lock';lock.write_text('stale owner')
        with self.assertRaisesRegex(RuntimeError,'lock'):self.calibrate(max_families=2)
        self.assertEqual(lock.read_text(),'stale owner')
        with self.assertRaises(ValueError):self.load()
        lock.unlink()
        annotation=self.review/'annotations-ko.json';value=quality._read(annotation)
        value['annotations'][2]['reason']='Changed provisional evidence';train._write(annotation,value)
        with self.assertRaises(ValueError):self.load()
        with self.assertRaises(ValueError):self.calibrate(max_families=2)

    def test_frozen_model_rubric_template_and_original_rows_cannot_be_rebound(self):
        self.calibrate(max_families=1)
        config_path=self.output/'run-config.json';original=config_path.read_bytes()
        mutations=[lambda c:c['generation'].update(repetition_penalty=1.1),
                   lambda c:c['settings'].update(model='another/model'),
                   lambda c:c.update(rubric=c['rubric']+' relaxed'),
                   lambda c:c.update(template=c['template']+' changed'),
                   lambda c:c['checkpoint_files'].update(**{'weights.safetensors':'changed'})]
        for mutate in mutations:
            config=quality._read(config_path);mutate(config);train._write(config_path,config)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.load()
            with self.assertRaises(ValueError):self.calibrate(max_families=2)
            config_path.write_bytes(original)
        path=self.output/'inputs.jsonl';items=quality._rows(path)
        items[0]['rows'][0]['provenance']['original_instruction']='Rewritten original provenance'
        self.write_rows(path,items)
        with self.assertRaises(ValueError):self.load()
        with self.assertRaises(ValueError):self.calibrate(max_families=2)

    def test_duplicate_or_reordered_whole_family_journal_and_extra_keys_fail(self):
        self.calibrate(max_families=2)
        path=self.output/'families.jsonl';records=quality._rows(path);original=path.read_bytes()
        for changed in ([records[0],records[0]],list(reversed(records)),
                        [dict(records[0],unexpected=True),records[1]]):
            self.write_rows(path,changed);self.rehash()
            with self.assertRaises(ValueError):self.load()
            with self.assertRaises(ValueError):self.calibrate(max_families=3)
            path.write_bytes(original);self.rehash()

    def test_nonfinite_or_corrupt_raw_generation_fails_closed_and_records_failure(self):
        original=self.fake_generate
        def corrupt(*args,**kwargs):
            evidence=original(*args,**kwargs)
            evidence[0]['raw_generated_ids'][-1]=0
            return evidence
        with patch.object(quality.runtime,'generate_batch',side_effect=corrupt):
            with self.assertRaises(ValueError):self.calibrate(max_families=1)
        loaded=self.load()
        self.assertEqual(loaded['report']['status'],'failed')
        self.assertEqual(loaded['report']['families_reviewed'],0)
        self.assertEqual(loaded['report']['accepted_families'],0)
        self.assertFalse(loaded['report']['computational_gates_met'])
        self.calibrate(max_families=1)
        self.assertEqual(self.load()['report']['status'],'incomplete')

    def test_legacy_report_success_cannot_be_invented_or_gates_called_human_validation(self):
        self.calibrate(max_families=1)
        path=self.output/'report.json';report=quality._read(path)
        report.update(status='complete',human_review='complete',suitability='validated')
        train._write(path,report);self.rehash()
        with self.assertRaises(ValueError):self.load()

    def test_cli_calibrate_and_verify_use_explicit_bound_inputs(self):
        import contextlib
        import io
        args=['--prepared',str(self.prepared),'--review-evidence',str(self.review),'--output',str(self.output)]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            report=quality.main(['calibrate',*args,'--max-families','1'])
        self.assertEqual(json.loads(output.getvalue())['families_reviewed'],1)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            verified=quality.main(['verify',*args])
        self.assertEqual(report,verified)
        self.assertEqual(json.loads(output.getvalue())['human_review'],'pending')

    def test_v2_explicit_output_contract_preserves_v1_bytes_and_semantic_rubric(self):
        import hashlib
        rows=[dict(lang=lang,instruction='Classify this.',response='Answer in '+lang)
              for lang in ('en','ko','zh')]
        v1=quality.family_messages(rows)
        self.assertEqual(hashlib.sha256(quality._json(v1).encode()).hexdigest(),
                         '6fc3a552bfc6bc4b94cb415f21c92ccbee6c95967a705d1f4de66775f53d09fb')
        v2=quality.family_messages(rows,rubric_version=2)
        self.assertTrue(v2[0]['content'].startswith(v1[0]['content']))
        contract=v2[0]['content'].split('END QUOTED FAMILY DATA',1)[1]
        for phrase in ('five top-level keys','en, ko, zh, equivalent_tasks, equivalence_reason',
                       'only quality and reason','only at the root','80 characters','no Markdown'):
            self.assertIn(phrase,contract)
        self.assertEqual(v2[0]['content'].count(quality.RUBRIC),1)
        # Version2 clarifies output layout, never repairs the same invalid output.
        malformed='```json\n'+json.dumps(judgment())+'\n```'
        self.assertFalse(quality.parse_judgment(malformed,finish_reason='eos')['family_pass'])

    def test_version2_frozen_selection_resume_and_reader_inference(self):
        import inspect
        self.assertIn('rubric_version',inspect.signature(quality.calibrate).parameters, 'Explicit prompt version selection is missing')
        self.calibrate(max_families=1,rubric_version=2)
        loaded=self.load()
        self.assertEqual(loaded['config']['settings']['rubric_version'],2)
        self.assertIn('80 characters',loaded['families'][0]['messages'][0]['content'])
        before={p.relative_to(self.output):p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError,'binding changed'):self.calibrate(max_families=2)
        self.assertEqual(before,{p.relative_to(self.output):p.read_bytes() for p in self.output.rglob('*') if p.is_file()})
        self.calibrate(max_families=2,rubric_version=2)
        self.assertEqual(len(self.load()['rows']),2)

    def test_v1_archive_reader_uses_frozen_version_but_cannot_append_changed_source(self):
        self.calibrate(max_families=1)
        before=self.load()
        sources=quality._source_paths();changed=self.root/'v2-implementation.py'
        changed.write_bytes(sources['quality.py'].read_bytes()+b'\n# New executing version.\n')
        with patch.object(quality,'_source_paths',return_value={**sources,'quality.py':changed}):
            loaded=self.load()
            self.assertEqual(loaded,before)
            self.assertEqual(loaded['config']['settings']['rubric_version'],1)
            with self.assertRaisesRegex(ValueError,'Current implementation source'):
                self.calibrate(max_families=2)
            with self.assertRaisesRegex(ValueError,'binding changed'):
                self.calibrate(max_families=2,rubric_version=2)

    def test_invalid_versions_and_version_relabeling_cannot_change_bound_prompts(self):
        for version in (True,False,0,3,'2',None):
            with self.subTest(version=version),self.assertRaises(ValueError):
                self.calibrate(max_families=1,rubric_version=version)
        self.assertFalse(self.output.exists())
        self.calibrate(max_families=1)
        config_path=self.output/'run-config.json';config=quality._read(config_path)
        config['settings']['rubric_version']=2;train._write(config_path,config)
        with self.assertRaises(ValueError):self.load()
        with self.assertRaises(ValueError):self.calibrate(max_families=2,rubric_version=2)

    def test_cli_explicit_v2_and_verify_infers_version_without_selection_argument(self):
        import contextlib
        import io
        args=['--prepared',str(self.prepared),'--review-evidence',str(self.review),'--output',str(self.output)]
        with contextlib.redirect_stdout(io.StringIO()):
            quality.main(['calibrate',*args,'--max-families','1','--rubric-version','2'])
            quality.main(['verify',*args])
        self.assertEqual(self.load()['config']['settings']['rubric_version'],2)


    def test_qwen_frozen_profile_reader_inference_and_cross_profile_resume_rejection(self):
        from transformers import AutoTokenizer
        from experiments.tiny_aya.guard import _snapshot
        snapshot, files = _snapshot('Qwen/Qwen3-14B', '40c069824f4251a91eefaf281ebe4c544efd3e18', False)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        with patch.object(quality, '_tokenizer', return_value=(self.tokenizer, files)):
            self.calibrate(max_families=1, rubric_version=2, judge_profile='qwen3-14b')
            loaded = self.load()
            config = loaded['config'];settings = config['settings']
            self.assertEqual(settings['judge_profile'], 'qwen3-14b')
            self.assertEqual(settings['model'], 'Qwen/Qwen3-14B')
            self.assertEqual(settings['revision'], '40c069824f4251a91eefaf281ebe4c544efd3e18')
            self.assertEqual(settings['mps_memory_fraction'], .9)
            self.assertEqual(settings['chat_template_kwargs'], dict(tokenize=True, add_generation_prompt=True,
                return_dict=False, truncation=False, enable_thinking=False))
            self.assertTrue(loaded['rows'][0]['judgment']['family_pass'])
            before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
            with self.assertRaises(ValueError): self.calibrate(max_families=2, rubric_version=2)
            self.assertEqual(before, {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()})
            self.calibrate(max_families=2, rubric_version=2, judge_profile='qwen3-14b')
            self.assertEqual(len(self.load()['rows']), 2)
            path = self.output/'run-config.json';original = path.read_bytes()
            mutations = [lambda c: c['settings'].update(judge_profile='aya-expanse-8b'),
                lambda c: c['settings'].update(revision=quality.REVISION),
                lambda c: c['settings']['chat_template_kwargs'].update(enable_thinking=True),
                lambda c: c['settings']['chat_template_kwargs'].pop('return_dict'),
                lambda c: c['settings'].update(mps_memory_fraction=.8),
                lambda c: c['generation'].update(bos_token_id=None)]
            for mutate in mutations:
                value = quality._read(path);mutate(value);train._write(path, value);self.rehash()
                with self.subTest(mutate=mutate), self.assertRaises(ValueError): self.load()
                with self.assertRaises(ValueError): self.calibrate(max_families=3, rubric_version=2, judge_profile='qwen3-14b')
                path.write_bytes(original);self.rehash()
            sources = quality._source_paths();changed = self.root/'changed-qwen.py'
            changed.write_bytes(sources['quality.py'].read_bytes()+b'\n# Changed implementation.\n')
            with patch.object(quality, '_source_paths', return_value={**sources, 'quality.py': changed}):
                self.assertEqual(len(self.load()['rows']), 2)
                with self.assertRaisesRegex(ValueError, 'Current implementation source'):
                    self.calibrate(max_families=3, rubric_version=2, judge_profile='qwen3-14b')

    def test_qwen_cli_selection_and_invalid_profile_fail_before_output_mutation(self):
        import contextlib
        import io
        for profile in (None, True, 'qwen3', 'other'):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.calibrate(max_families=1, judge_profile=profile)
        self.assertFalse(self.output.exists())
        args = ['--prepared', str(self.prepared), '--review-evidence', str(self.review), '--output', str(self.output)]
        with patch.object(quality, 'calibrate', return_value={'status': 'incomplete'}) as run, contextlib.redirect_stdout(io.StringIO()):
            quality.main(['calibrate', *args, '--rubric-version', '2', '--judge-profile', 'qwen3-14b'])
        self.assertEqual(run.call_args.kwargs, dict(rubric_version=2, judge_profile='qwen3-14b'))


class JudgeProfileTests(unittest.TestCase):
    def test_qwen_profile_selection_and_canonical_generation(self):
        import inspect
        self.assertIn('judge_profile', inspect.signature(quality.calibrate).parameters,
                      'Explicit pinned judge profile selection is missing')
        config = quality.generation_config(judge_profile='qwen3-14b')
        self.assertEqual(config.eos_token_id, [151645, 151643])
        self.assertEqual(config.pad_token_id, 151643)
        self.assertEqual(config.bos_token_id, 151643)
        self.assertEqual(config.max_new_tokens, 384)
        self.assertFalse(config.do_sample)
        with self.assertRaises(ValueError): quality.generation_config(judge_profile='unknown')

    def test_qwen_official_template_exact_ids_and_empty_thinking_prefix(self):
        from transformers import AutoTokenizer
        from experiments.tiny_aya.guard import _snapshot
        snapshot, _ = _snapshot('Qwen/Qwen3-14B', '40c069824f4251a91eefaf281ebe4c544efd3e18', False)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.assertEqual((len(tokenizer), tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id),
                         (151669, 151645, 151643, None))
        rows = [dict(lang=lang, instruction='Say hello.', response='Hello.') for lang in ('en','ko','zh')]
        data = dict(families=[dict(family_id='f', rows=rows)])
        frozen = quality._freeze_inputs(data, tokenizer, rubric_version=2, judge_profile='qwen3-14b')[0]
        messages = quality.family_messages(rows, rubric_version=2)
        self.assertEqual(frozen['messages'], messages)
        expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=False, truncation=False, enable_thinking=False)
        self.assertEqual(frozen['input_ids'], expected)
        self.assertTrue(tokenizer.decode(expected, skip_special_tokens=False).endswith(
            '<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        enabled = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=False, truncation=False, enable_thinking=True)
        self.assertNotEqual(expected, enabled)
        generation = quality.generation_config(judge_profile='qwen3-14b')
        for eos in (151645, 151643):
            raw = tokenizer.encode(json.dumps(judgment()), add_special_tokens=False) + [eos, 151643]
            evidence = runtime.completion_evidence(tokenizer, expected, raw, generation_config=generation,
                padded_width=len(expected), elapsed_batch_seconds=0)
            self.assertTrue(quality.parse_judgment(evidence['response'], finish_reason=evidence['finish_reason'])['family_pass'])

    def test_qwen_mps_budget_is_explicit_and_aya_legacy_budget_unchanged(self):
        with patch('torch.backends.mps.is_available', return_value=True), \
             patch('torch.mps.set_per_process_memory_fraction') as fraction:
            self.assertEqual(quality._device(judge_profile='qwen3-14b'), 'mps')
            fraction.assert_called_once_with(.9)
        with patch('torch.backends.mps.is_available', return_value=True), \
             patch('torch.mps.set_per_process_memory_fraction') as fraction:
            self.assertEqual(quality._device(), 'mps')
            fraction.assert_called_once_with(.8)

    def test_qwen_direct_cpu_loader_validates_architecture_attention_and_every_tensor(self):
        import torch
        from types import SimpleNamespace
        class Fixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type='qwen3', architectures=['Qwen3ForCausalLM'],
                    num_hidden_layers=40, hidden_size=5120, vocab_size=151936,
                    num_attention_heads=40, num_key_value_heads=8, head_dim=128, _attn_implementation='eager')
                self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(40)])
                for layer in self.layers:
                    layer.self_attn = torch.nn.Module();layer.self_attn.config = self.config
                self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            def set_attn_implementation(self, value): self.config._attn_implementation = value
            def _prepare_generation_config(self, config): return config, {}
            def to(self, *args, **kwargs): raise AssertionError('Direct device loader must not duplicate via .to()')
        generation = quality.generation_config(judge_profile='qwen3-14b')
        model = Fixture()
        with patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model) as loader:
            loaded = quality._new_model(Path('/fixture'), 'cpu', generation, judge_profile='qwen3-14b')
        self.assertIs(loaded, model)
        self.assertEqual(loader.call_args.kwargs, dict(local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='eager', device_map='cpu'))
        self.assertFalse(model.training);self.assertFalse(model.weight.requires_grad)
        changes = [lambda m: setattr(m.config, 'num_key_value_heads', 40),
            lambda m: setattr(m.config, 'head_dim', 64), lambda m: setattr(m.config, 'vocab_size', 151669),
            lambda m: setattr(m.config, 'architectures', ['OtherModel']), lambda m: m.layers.pop(-1),
            lambda m: setattr(m, 'peft_config', {'adapter': True}),
            lambda m: setattr(m, 'weight', torch.nn.Parameter(torch.ones(2))),
            lambda m: m.weight.data.fill_(float('nan')),
            lambda m: setattr(m.weight, 'grad', torch.ones_like(m.weight)),
            lambda m: setattr(m, '_hf_peft_config_loaded', True),
            lambda m: setattr(m, 'weight', torch.nn.Parameter(torch.empty(2, device='meta', dtype=torch.bfloat16))),
            lambda m: m.layers[0].register_parameter('lora_A', torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)))]
        for change in changes:
            model = Fixture();change(model)
            with self.subTest(change=change), patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model):
                with self.assertRaises(ValueError):
                    quality._new_model(Path('/fixture'), 'cpu', generation, judge_profile='qwen3-14b')
