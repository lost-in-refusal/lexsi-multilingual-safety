"""CPU-only synthetic tests for public source preparation."""
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.tiny_aya import prepare


LANGS = ('en', 'ko', 'zh')


def bactrian_family(identity, text=None):
    # Unique alphabets/wording make incidental near-duplicates explicit in each test.
    words = text or {'en': 'Describe a rainbow and its colors',
                     'ko': '무지개가 생기는 이유를 설명해주세요',
                     'zh': '请说明彩虹的颜色及其形成原因'}
    return {lang: {'id': identity, 'instruction': words[lang], 'input': '',
                   'output': {'en': 'A rainbow has several colors.',
                              'ko': '무지개에는 여러 가지 색깔이 있습니다.',
                              'zh': '彩虹包含多种颜色，是折射形成的。'}[lang]}
            for lang in LANGS}


def columns(*families):
    return {lang: [f[lang] for f in families if lang in f] for lang in LANGS}


def jail_rows(count=126):
    rows = []
    for i in range(count):
        rows.append({'id': str(i), 'source': 'synthetic-public-fixture',
                     'tags': "['category-a', 'category-b']" if i % 2 else "['category-a']",
                     **{lang: f'{lang} synthetic request {i}' for lang in
                        ('en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv')}})
    return rows


class CandidateTests(unittest.TestCase):
    def test_whole_family_filters_prevent_incomplete_multilingual_output(self):
        good = bactrian_family('good')
        broken = []
        for i, change in enumerate(({'input': 'context'}, {'instruction': ' x y '},
                                    {'output': '抱歉，我无法回答。'},
                                    {'instruction': 'As an AI I cannot respond'},
                                    {'output': 'Here is <|im_end|> content'},
                                    {'output': '죄송합니다. 답할 수 없습니다.'})):
            family = bactrian_family(f'bad-{i}')
            family['ko'].update(change)
            broken.append(family)
        missing = bactrian_family('missing'); del missing['zh']
        rows, stats = prepare.select_candidates(columns(good, *broken, missing), 1)
        self.assertEqual([r['lang'] for r in rows], ['en', 'ko', 'zh'])
        self.assertEqual({r['family_id'] for r in rows}, {'bactrian:good'})
        self.assertEqual(stats['selected_families'], 1)
        self.assertEqual(stats['rejected_incomplete_families'], 1)
        self.assertEqual(stats['rejected_content_families'], 6)

    def test_explicit_null_empty_and_whitespace_inputs_mean_no_supplied_context(self):
        for inputs in ((None, '', ' \t\n'), ('', None, ' '), (' ', '', None)):
            with self.subTest(inputs=inputs):
                family = bactrian_family('mixed-empty-input')
                for lang, value in zip(LANGS, inputs):
                    family[lang]['input'] = value
                rows, stats = prepare.select_candidates(columns(family), 1)
                self.assertEqual([row['lang'] for row in rows], ['en', 'ko', 'zh'])
                self.assertEqual({row['family_id'] for row in rows}, {'bactrian:mixed-empty-input'})
                self.assertEqual(stats['rejected_content_families'], 0)

    def test_missing_wrong_type_and_nonempty_input_reject_whole_family(self):
        good = bactrian_family('valid-empty-input')
        broken = []
        for lang in LANGS:
            for label, value in (('missing', None), ('number', 0), ('boolean', False),
                                 ('list', []), ('dict', {}), ('supplied', ' context ')):
                family = bactrian_family(f'{lang}-{label}')
                if label == 'missing':
                    del family[lang]['input']
                else:
                    family[lang]['input'] = value
                broken.append(family)
        rows, stats = prepare.select_candidates(columns(good, *broken), 1)
        self.assertEqual(len(rows), 3)
        self.assertEqual({row['family_id'] for row in rows}, {'bactrian:valid-empty-input'})
        self.assertEqual(stats['rejected_content_families'], 18)

    def test_cleaner_changes_have_original_instruction_and_verbatim_response(self):
        family = bactrian_family('normalization')
        family['en']['instruction'] = '  Describe cafe\u0301 colors\x00  '
        family['en']['output'] = '  Preserve cafe\u0301 and  spaces\nexactly.  '
        rows, stats = prepare.select_candidates(columns(family), 1)
        en = rows[0]
        self.assertEqual(en['instruction'], 'Describe café colors')
        self.assertEqual(en['response'], 'Preserve cafe\u0301 and  spaces\nexactly.')
        self.assertEqual(en['provenance']['original_instruction'], 'Describe cafe\u0301 colors\x00')
        self.assertEqual(stats['cleaner_changed_instructions'], 1)
        self.assertEqual([p['step_name'] for p in en['provenance']['steps']],
                         ['TextCleaner', 'ExactDeduplicator'])
        self.assertNotIn('timestamp', en['provenance']['steps'][0])

    def test_global_exact_and_near_duplicates_exclude_whole_later_family(self):
        first = bactrian_family('one')
        exact = bactrian_family('two')
        exact['ko']['instruction'] = '다른 언어의 유효한 고유한 지시사항입니다'
        near = bactrian_family('three')
        near['en']['instruction'] += '!'
        near['ko']['instruction'] = '한글로 작성된 별개의 새 요청을 처리해주세요'
        near['zh']['instruction'] = '请根据下面的材料完成独立的不同任务'
        unique = bactrian_family('four', {'en': 'Calculate how many sides a square has',
                                         'ko': '정사각형에 있는 변의 개수는 몇개인가요',
                                         'zh': '一个正方形总共有多少条边，请给出答案'})
        data = columns(first, exact, near, unique)
        rows, stats = prepare.select_candidates(data, 2)
        ids = {r['family_id'] for r in rows}
        self.assertIn('bactrian:four', ids)
        self.assertEqual(len(ids & {'bactrian:one', 'bactrian:two', 'bactrian:three'}), 1)
        self.assertEqual(len(rows), 6)
        self.assertGreaterEqual(stats['rejected_exact_duplicate_families'] +
                                stats['rejected_near_duplicate_families'], 2)

    def test_prefix_index_keeps_threshold_matches_and_rejects_lower_similarity(self):
        base = frozenset(f'gram-{i}' for i in range(20))
        threshold_match = frozenset(f'gram-{i}' for i in range(17))
        below_threshold = frozenset(f'gram-{i}' for i in range(16))
        index = prepare._NearIndex([base, threshold_match, below_threshold])
        index.add(base)
        self.assertTrue(index.matches(threshold_match))  # 17 / 20 == 0.85
        self.assertFalse(index.matches(below_threshold))  # 16 / 20 == 0.80

    def test_duplicate_source_id_rejects_whole_family(self):
        one = bactrian_family('one'); data = columns(one)
        data['en'].append(copy.deepcopy(data['en'][0]))
        with self.assertRaisesRegex(ValueError, 'Insufficient.*0'):
            prepare.select_candidates(data, 1)

    def test_insufficient_count_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient.*1.*2'):
            prepare.select_candidates(columns(bactrian_family('one')), 2)

    def test_selection_uses_seeded_hash_and_ignores_input_order(self):
        family = bactrian_family('first')
        second = bactrian_family('second', {'en': 'How does a bicycle wheel rotate?',
                                           'ko': '자전거의 바퀴는 어떤 원리로 회전합니까',
                                           'zh': '自行车的轮子为什么能够不断地转动呢'})
        data = columns(family, second)
        expected = min(('first', 'second'), key=lambda s: hashlib.sha256(f'42:{s}'.encode()).hexdigest())
        rows, _ = prepare.select_candidates(data, 1)
        self.assertEqual({r['family_id'] for r in rows}, {'bactrian:' + expected})
        reverse = {lang: list(reversed(values)) for lang, values in data.items()}
        self.assertEqual(rows, prepare.select_candidates(reverse, 1)[0])


class MultiJailTests(unittest.TestCase):
    def test_attack_alignment_disjoint_and_review_is_pending_matched_pairs(self):
        attacks, alignment, review, stats = prepare.select_multijail(jail_rows())
        self.assertEqual(len(attacks), 400)
        self.assertEqual(len(alignment), 72)
        self.assertEqual(len(review), 40)
        self.assertEqual({r['lang'] for r in attacks}, {'en', 'ko', 'zh', 'sw'})
        self.assertEqual({r['lang'] for r in alignment}, {'en', 'ko', 'zh'})
        self.assertFalse({r['family_id'] for r in attacks} & {r['family_id'] for r in alignment})
        lookup = {(r['family_id'], r['lang']): r for r in attacks}
        for row in review:
            self.assertEqual(row['english_instruction'], lookup[(row['family_id'], 'en')]['instruction'])
            self.assertEqual(row['translated_instruction'], lookup[(row['family_id'], row['lang'])]['instruction'])
            self.assertEqual(row['review_status'], 'pending')
            self.assertIsNone(row['human_label'])
        self.assertEqual(stats['attack_category_family_counts']['category-a'], 100)
        self.assertEqual(stats['alignment_category_family_counts']['category-a'], 24)
        self.assertTrue(all(r['source'] == 'synthetic-public-fixture' for r in attacks))
        self.assertTrue(all('response' not in r for r in attacks + alignment))
        self.assertEqual((attacks, alignment, review, stats), prepare.select_multijail(list(reversed(jail_rows()))))

    def test_malformed_tags_are_rejected_without_execution(self):
        for tags in ("__import__('os').system('false')", '{broken', "['ok', 1]", '{}'):
            rows = jail_rows(124); rows[0]['tags'] = tags
            with self.subTest(tags=tags), self.assertRaisesRegex(ValueError, 'tags'):
                prepare.select_multijail(rows)

    def test_incomplete_and_duplicate_ids_cannot_enter_either_partition(self):
        rows = jail_rows(126); rows[0]['sw'] = ''; rows[1]['en'] = ''
        rows.append(copy.deepcopy(rows[2]))
        with self.assertRaisesRegex(ValueError, 'Insufficient.*123.*124'):
            prepare.select_multijail(rows)


class BundleTests(unittest.TestCase):
    def source_files(self, root):
        paths = {}
        for lang, values in columns(bactrian_family('only')).items():
            paths[lang] = root / f'{lang}.json.gz'
            paths[lang].write_bytes(gzip.compress(json.dumps(values, ensure_ascii=False).encode(), mtime=0))
        paths['multijail'] = root / 'MultiJail.csv'
        rows = jail_rows()
        with paths['multijail'].open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        return paths

    def test_bundle_is_byte_stable_and_verification_rejects_tamper_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = self.source_files(root)
            one = root / 'results' / 'tiny-aya' / 'one'
            two = root / 'results' / 'tiny-aya' / 'two'
            manifest = prepare.prepare_from_files(one, paths, candidate_families=1)
            prepare.prepare_from_files(two, paths, candidate_families=1)
            self.assertEqual({p.name: p.read_bytes() for p in one.iterdir()},
                             {p.name: p.read_bytes() for p in two.iterdir()})
            self.assertNotIn(str(root), json.dumps(manifest))
            self.assertEqual(prepare.verify_bundle(one), manifest)
            self.assertEqual(manifest['counts']['candidates'], 3)
            self.assertEqual(manifest['counts']['attacks'], 400)
            self.assertEqual(set(manifest['source_files']), {'en', 'ko', 'zh', 'multijail'})
            with self.assertRaises(FileExistsError):
                prepare.prepare_from_files(one, paths, candidate_families=1)
            (one / 'candidates.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                prepare.verify_bundle(one)

    def test_verify_checks_family_structure_even_if_digest_is_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = self.source_files(root)
            output = root / 'results' / 'tiny-aya' / 'bundle'
            manifest = prepare.prepare_from_files(output, paths, candidate_families=1)
            target = output / 'alignment.jsonl'
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            rows[0]['lang'] = 'sw'
            target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            manifest['files']['alignment.jsonl']['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
            (output / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'language|structure'):
                prepare.verify_bundle(output)

    def test_invalid_count_is_rejected_before_any_network_access(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'results' / 'tiny-aya' / 'invalid'
            with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('network access')):
                for count in (0, -1, True, 1.5):
                    with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'positive integer'):
                        prepare.prepare_sources(output, candidate_families=count)

    def test_verifier_rejects_inconsistent_family_categories_and_provenance(self):
        for name, field, replacement in (
            ('attacks.jsonl', 'harm_categories', ['invented-category']),
            ('candidates.jsonl', 'provenance', {'original_instruction': 'changed',
                                              'instruction_changed': False, 'steps': []}),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); paths = self.source_files(root)
                output = root / 'results' / 'tiny-aya' / 'bundle'
                manifest = prepare.prepare_from_files(output, paths, candidate_families=1)
                target = output / name
                rows = [json.loads(line) for line in target.read_text().splitlines()]
                rows[81 if name == 'attacks.jsonl' else 1][field] = replacement
                target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
                manifest['files'][name]['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
                (output / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'category|provenance'):
                    prepare.verify_bundle(output)

    def test_verify_rejects_changed_config_and_source_pins(self):
        for field in ('config', 'source_files'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); paths = self.source_files(root)
                output = root / 'results' / 'tiny-aya' / 'bundle'
                manifest = prepare.prepare_from_files(output, paths, candidate_families=1)
                if field == 'config':
                    manifest[field]['seed'] = 43
                else:
                    manifest[field]['en']['revision'] = '0' * 40
                (output / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'pin|Configuration'):
                    prepare.verify_bundle(output)

    def test_output_outside_ignored_results_is_refused_before_download(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'results/tiny-aya'):
                prepare.prepare_sources(Path(temp) / 'publishable', candidate_families=1)


if __name__ == '__main__':
    unittest.main()
