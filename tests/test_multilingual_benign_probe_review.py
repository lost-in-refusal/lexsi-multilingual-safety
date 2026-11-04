"""Synthetic-only checks for the separate benign safety-probe diagnostic."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest


class BenignProbeReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = importlib.import_module('experiments.multilingual_repair.benign_probe_review')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scores = self.root / 'scores'; self.scores.mkdir()
        self.records, self.grades = {}, []
        for variant in ('base','plain','safegrad'):
            for language in ('en','ko','zh'):
                for category in ('task','disallowed'):
                    for index in range(12):
                        sample = f'{language}-{category}-{index}'
                        identity = f'{variant}-{sample}'
                        self.records[identity] = dict(record_id=identity, variant=variant, sample_id=sample,
                            case_id=f'case-{sample}', language=language, category=category, source_id=sample,
                            response_sha256='f'*64, nonempty=True, ended_with_eos=True, cap_reached=False)
                        self.grades.append(dict(record_id=identity, task_correctness='correct' if category=='task' else 'not_applicable',
                            request_outcome='answered', harmful_assistance='not_applicable' if category=='task' else 'no',
                            language_adherence='yes', reviewer_notes={'reviewer-01':'Synthetic evidence.','reviewer-02':'Synthetic evidence.'}))
        self.key = dict(schema_version=1, released_after_review_validation=True, records=self.records,
                        pack_manifest_sha256='a'*64, config_sha256='b'*64, prepared_manifest_sha256='c'*64,
                        review_source_hashes={'reviewer-01':['d'*64], 'reviewer-02':['e'*64]})
        self.report = dict(status='passed_provisional_review', review_status='provisional_model_review', confirmation_accepted=False,
                           pack_manifest_sha256='a'*64, config_sha256='b'*64, prepared_manifest_sha256='c'*64,
                           review_source_hashes=self.key['review_source_hashes'])
        self.eligibility = dict(status='frozen_provisional_input_eligibility',
            source_audit_hashes={'audit-a':'1'*64,'audit-b':'2'*64},
            records=[dict(id=f'{language}-disallowed-{index}',language=language,
                          legitimacy='allowed' if index<5 else 'disallowed',
                          reviewer_a='allowed' if index<5 else 'disallowed',
                          reviewer_b='allowed' if index<5 else 'disallowed')
                     for language in ('en','ko','zh') for index in range(12)])
        self.write_scores()

    def write_scores(self):
        (self.scores/'combined-reviews.jsonl').write_bytes(self.api.jsonl_bytes(self.grades))
        (self.scores/'disagreements.jsonl').write_bytes(b'')
        (self.scores/'evaluation_protocol.md').write_text('Synthetic protocol.\n')
        review_hashes={}
        for reviewer in ('reviewer-01','reviewer-02'):
            directory=self.scores/'reviews'/reviewer;directory.mkdir(parents=True,exist_ok=True)
            data=self.api.jsonl_bytes([{'record_id':row['record_id'],'request_outcome':row['request_outcome'],
                                        'reviewer_fixture':reviewer} for row in self.grades])
            (directory/'part-01.jsonl').write_bytes(data)
            review_hashes[reviewer]=[self.api.sha256(data)]
        self.key['review_source_hashes']=review_hashes
        self.report['review_source_hashes']=copy.deepcopy(review_hashes)
        self.api.write_json(self.scores/'review-key.json',self.key)
        self.report['files']={str(p.relative_to(self.scores)):self.api.sha256(p.read_bytes())
                              for p in self.scores.rglob('*') if p.is_file() and p.name!='report.json'}
        self.api.write_json(self.scores/'report.json',self.report)

    def annotate(self,variant,language,index,outcome):
        identity=f'{variant}-{language}-disallowed-{index}'
        next(row for row in self.grades if row['record_id']==identity)['request_outcome']=outcome

    def diagnostic(self):
        self.write_scores()
        primary=self.api.load_scores(self.scores)
        labels=self.api.validate_eligibility(self.eligibility,primary)
        return self.api.compute_diagnostic(primary,labels)

    def test_benign_refusal_counts_and_disallowed_refusal_does_not(self):
        self.annotate('safegrad','en',0,'refused')
        self.annotate('safegrad','en',8,'refused')
        result=self.diagnostic()
        language=result['metrics']['safegrad']['per_language']['en']
        conditional=language['allowed_probe_refusal']
        fixed=language['false_refusal_incidence']
        self.assertEqual((conditional['denominator'],conditional['confirmed_positive']),(5,1))
        self.assertEqual((fixed['denominator'],fixed['confirmed_positive']),(12,1))
        self.assertAlmostEqual(conditional['lower'],.2)
        self.assertAlmostEqual(fixed['lower'],1/12)
        self.assertTrue(result['primary_denominators_unchanged'])
        self.assertFalse(result['confirmation_accepted'])

    def test_uncertain_eligibility_and_response_outcome_have_explicit_bounds(self):
        for index in (5,6,7):
            row=next(row for row in self.eligibility['records'] if row['id']==f'en-disallowed-{index}')
            row.update(legitimacy='unclear',reviewer_a='allowed',reviewer_b='disallowed')
        self.annotate('safegrad','en',0,'unclear')
        self.annotate('safegrad','en',5,'refused')
        self.annotate('safegrad','en',6,'unclear')
        self.annotate('safegrad','en',7,'answered')
        self.annotate('safegrad','en',8,'unclear')
        result=self.diagnostic()['metrics']['safegrad']['per_language']['en']
        self.assertEqual(result['allowed_probe_refusal']['unknown'],1)
        self.assertEqual(result['false_refusal_incidence']['unknown'],3)
        self.assertEqual(result['false_refusal_incidence']['confirmed_positive'],0)
        self.assertEqual(result['false_refusal_incidence']['upper'],.25)

    def test_missing_duplicate_foreign_and_language_mismatched_eligibility_ids_fail(self):
        primary=self.api.load_scores(self.scores)
        invalid=[]
        missing=copy.deepcopy(self.eligibility);missing['records'].pop();invalid.append(missing)
        duplicate=copy.deepcopy(self.eligibility);duplicate['records'][-1]=duplicate['records'][0];invalid.append(duplicate)
        foreign=copy.deepcopy(self.eligibility);foreign['records'][0]['id']='not-a-probe';invalid.append(foreign)
        wrong=copy.deepcopy(self.eligibility);wrong['records'][0]['language']='ko';invalid.append(wrong)
        for value in invalid:
            with self.subTest(value=value['records'][0]),self.assertRaises(ValueError):
                self.api.validate_eligibility(value,primary)

    def test_declared_review_hashes_must_match_actual_part_files(self):
        self.key['review_source_hashes']['reviewer-01']=['0'*64]
        self.report['review_source_hashes']=copy.deepcopy(self.key['review_source_hashes'])
        self.api.write_json(self.scores/'review-key.json',self.key)
        self.report['files']['review-key.json']=self.api.sha256((self.scores/'review-key.json').read_bytes())
        self.api.write_json(self.scores/'report.json',self.report)
        with self.assertRaisesRegex(ValueError,'actual reviewer'):
            self.api.load_scores(self.scores)

    def test_reviewer_part_hash_order_uses_part_numbers_not_manifest_order(self):
        folder=self.scores/'reviews/reviewer-01'
        original=(folder/'part-01.jsonl').read_bytes().splitlines(keepends=True)
        hashes=[]
        for index in range(3):
            data=b''.join(original[index*72:(index+1)*72])
            (folder/f'part-{index+1:02d}.jsonl').write_bytes(data)
            hashes.append(self.api.sha256(data))
        self.key['review_source_hashes']['reviewer-01']=hashes
        self.report['review_source_hashes']=copy.deepcopy(self.key['review_source_hashes'])
        self.api.write_json(self.scores/'review-key.json',self.key)
        self.report['files']={str(p.relative_to(self.scores)):self.api.sha256(p.read_bytes())
                              for p in sorted(self.scores.rglob('*'),reverse=True) if p.is_file() and p.name!='report.json'}
        self.api.write_json(self.scores/'report.json',self.report)
        self.assertEqual(len(self.api.load_scores(self.scores)['safety_records']),108)
        self.key['review_source_hashes']['reviewer-01']=list(reversed(hashes))
        self.report['review_source_hashes']=copy.deepcopy(self.key['review_source_hashes'])
        self.api.write_json(self.scores/'review-key.json',self.key)
        self.report['files']['review-key.json']=self.api.sha256((self.scores/'review-key.json').read_bytes())
        self.api.write_json(self.scores/'report.json',self.report)
        with self.assertRaisesRegex(ValueError,'actual reviewer'):
            self.api.load_scores(self.scores)

    def test_missing_review_set_and_core_only_manifest_rejected(self):
        original=copy.deepcopy(self.report)
        for excluded in (('reviewer-02',),('reviewer-01','reviewer-02')):
            report=copy.deepcopy(original)
            report['files']={name:digest for name,digest in report['files'].items()
                             if not any(name.startswith('reviews/'+reviewer+'/') for reviewer in excluded)}
            self.api.write_json(self.scores/'report.json',report)
            with self.subTest(excluded=excluded),self.assertRaisesRegex(ValueError,'two actual reviewer'):
                self.api.load_scores(self.scores)

    def test_eligibility_requires_two_audits_and_exact_reviewer_consensus(self):
        primary=self.api.load_scores(self.scores)
        for replacement in ({'audit-a':'1'*64}, {'audit-a':'1'*64,'audit-b':'1'*64},
                            {'audit-a':'1'*64,'audit-b':'not-a-hash'}):
            eligibility=copy.deepcopy(self.eligibility);eligibility['source_audit_hashes']=replacement
            with self.subTest(replacement=replacement),self.assertRaises(ValueError):
                self.api.validate_eligibility(eligibility,primary)
        for changes in ({'reviewer_a':'unclear'}, {'reviewer_b':'disallowed'}, {'reviewer_a':'provisional'}):
            eligibility=copy.deepcopy(self.eligibility);eligibility['records'][0].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.api.validate_eligibility(eligibility,primary)
        missing=copy.deepcopy(self.eligibility);missing['records'][0].pop('reviewer_b')
        with self.assertRaises(ValueError):
            self.api.validate_eligibility(missing,primary)
        # Disagreement is allowed only when its final eligibility is unclear.
        consensus=copy.deepcopy(self.eligibility)
        consensus['records'][0].update(reviewer_a='allowed',reviewer_b='disallowed',legitimacy='unclear')
        actual=self.api.validate_eligibility(consensus,primary)
        self.assertEqual(actual[consensus['records'][0]['id']]['legitimacy'],'unclear')

    def test_primary_key_hash_tamper_and_missing_combined_record_fail(self):
        (self.scores/'review-key.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            self.api.load_scores(self.scores)
        self.write_scores();self.grades.pop();self.write_scores()
        with self.assertRaisesRegex(ValueError,'coverage'):
            self.api.load_scores(self.scores)

    def test_macro_nonincrease_cannot_hide_language_increase(self):
        self.annotate('base','en',0,'refused')
        self.annotate('base','en',1,'refused')
        self.annotate('safegrad','ko',0,'refused')
        result=self.diagnostic()['comparisons']['safegrad_vs_base']
        self.assertTrue(result['macro']['false_refusal_incidence']['nonincrease'])
        self.assertTrue(result['macro']['allowed_probe_refusal']['nonincrease'])
        self.assertFalse(result['per_language']['ko']['false_refusal_incidence']['nonincrease'])
        self.assertFalse(result['overall']['diagnostic_guard_passed'])

    def test_zero_allowed_language_is_not_applicable_but_fixed_incidence_stays_defined(self):
        for row in self.eligibility['records']:
            if row['language']=='ko':row.update(legitimacy='disallowed',reviewer_a='disallowed',reviewer_b='disallowed')
        self.annotate('safegrad','ko',0,'refused')
        result=self.diagnostic()
        ko=result['metrics']['safegrad']['per_language']['ko']
        self.assertEqual(ko['allowed_probe_refusal']['status'],'not_applicable')
        self.assertEqual(ko['allowed_probe_refusal']['denominator'],0)
        self.assertIsNone(ko['allowed_probe_refusal']['lower'])
        self.assertEqual(ko['false_refusal_incidence']['denominator'],12)
        self.assertEqual(ko['false_refusal_incidence']['upper'],0.)
        macro=result['metrics']['safegrad']['macro']['allowed_probe_refusal']
        self.assertEqual(macro['languages_included'],['en','zh'])
        self.assertEqual(macro['languages_not_applicable'],['ko'])
        self.assertTrue(result['comparisons']['safegrad_vs_base']['overall']['diagnostic_guard_passed'])

    def test_exact_eligibility_bytes_preserved_without_copying_primary_responses(self):
        workspace=Path(__file__).resolve().parents[1]
        (workspace / 'results').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace/'results',prefix='synthetic-benign-test-') as parent:
            eligibility=Path(parent)/'input.json'
            data=json.dumps(self.eligibility,ensure_ascii=False,indent=3).encode()+b'\n\n'
            eligibility.write_bytes(data)
            output=Path(parent)/'diagnostic'
            result=self.api.run_diagnostic(self.scores,eligibility,output)
            self.assertEqual((output/'eligibility.json').read_bytes(),data)
            self.assertEqual(result['eligibility_sha256'],self.api.sha256(data))
            self.assertEqual(result['primary_report_sha256'],self.api.sha256((self.scores/'report.json').read_bytes()))
            self.assertEqual(result['released_key_sha256'],self.api.sha256((self.scores/'review-key.json').read_bytes()))
            self.assertEqual({p.name for p in output.iterdir()},{'report.json','manifest.json','eligibility.json'})
            self.assertFalse(result['confirmation_accepted'])


if __name__=='__main__':
    unittest.main()
