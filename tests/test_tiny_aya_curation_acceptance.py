"""Independent source acceptance: real triage/guard readers, synthetic CPU runtimes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import test_tiny_aya_curate as triage_fixtures
import test_tiny_aya_guard as safety_fixtures
import test_tiny_aya_prepare as source_fixtures
from experiments.tiny_aya import prepare, guard, quality, curate, train
try:
    from experiments.tiny_aya import curation_acceptance as acceptance
except ImportError:
    acceptance = None

REAL_VERIFY = prepare.verify_bundle


def valid_pool(fixture, path, rows):
    source = prepare.CONFIG['sources']['bactrian']
    for row in rows:
        row.update(source=source['repo'], source_revision=source['revision'])
        row['provenance']['steps'] = [dict(step_name=n, step_version='fixture', config_hash='a'*64)
                                      for n in ('TextCleaner', 'ExactDeduplicator')]
    attacks, alignment, review, stats = prepare.select_multijail(source_fixtures.jail_rows())
    files = dict(zip(prepare.DATA_FILES, (rows, attacks, alignment, review)))
    for name, values in files.items(): fixture.write_rows(path/name, values)
    train._write(path/'config.json', prepare.CONFIG)
    sources = {}
    for lang in prepare.LANGUAGES:
        sources[lang] = dict(repo=source['repo'], revision=source['revision'], filename=source['files'][lang], sha256='a'*64)
    jail = prepare.CONFIG['sources']['multijail']
    sources['multijail'] = {k:jail[k] for k in ('repo','revision','filename')} | dict(sha256='b'*64)
    train._write(path/'manifest.json', dict(schema_version=1, status=prepare.CONFIG['status'], config=prepare.CONFIG,
        config_sha256=train._hash_file(path/'config.json'), candidate_families=len(rows)//3, source_files=sources,
        counts={n[:-6]:len(v) for n,v in files.items()}, candidate_statistics=dict(selected_families=len(rows)//3,
        selected_changed_instructions=0), multijail_statistics=stats, curation={'fixture':'curator'},
        files={n:dict(sha256=train._hash_file(path/n),rows=len(v)) for n,v in files.items()}))


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(acceptance, 'Task10 acceptance module is missing')
        fixture = triage_fixtures.CurateTests(methodName='runTest')
        fixture.write_pool = types.MethodType(valid_pool, fixture)
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.f = fixture; self.root = fixture.root
        self.enterContext(patch.object(prepare, 'verify_bundle', REAL_VERIFY))
        self.enterContext(patch.object(guard, 'verify_bundle', REAL_VERIFY))
        self.safety = safety_fixtures.Runtime()
        self.enterContext(patch.object(guard, '_Runtime', return_value=self.safety))
        self.prepared = self.root/'accepted-candidates'
        self.screened = self.root/'results'/'tiny-aya'/'fresh-safety'
        self.audit = self.root/'audit'

    def freeze(self):
        self.f.score(None)
        return acceptance.freeze(self.f.pool, self.f.output, self.prepared)

    def audit_pack(self):
        self.freeze()
        guard.screen_candidates(self.prepared, self.screened)
        return acceptance.freeze_audit(self.prepared, self.screened, self.audit)

    def annotate(self, directory=None, labels=None):
        directory = directory or self.audit
        value = quality._read(directory/'annotations.template.json')
        value.update(status='complete', pack_manifest_sha256=train._hash_file(directory/'manifest.json'), reviewer=dict(name='Independent CPU reviewer', model_or_version='not_available',
                     method='Read each original request and response independently; source-grounded review.'))
        for index, row in enumerate(value['annotations']):
            row.update(quality=(labels or {}).get(index, 'usable'), reason='Response addresses the original request.')
            if row['quality']=='incorrect':
                row.update(reason='Material answer defect in the quoted response.', source_evidence=[dict(field='response', quote='Hello')])
        train._write(directory/'annotations.json', value)

    def test_freeze_is_prepared_compatible_and_original_rows_are_immutable(self):
        self.freeze()
        self.assertEqual(REAL_VERIFY(self.prepared)['candidate_families'],900)
        self.assertEqual(guard._rows(self.prepared/'candidates.jsonl'), self.f.originals[90:2790])
        for name in curate.AUXILIARY_FILES:
            self.assertEqual((self.prepared/name).read_bytes(),(self.f.pool/name).read_bytes())
        self.assertEqual(acceptance.verify_bundle(self.prepared)['candidate_families'],900)
        manifest=quality._read(self.prepared/'manifest.json')
        rows=guard._rows(self.prepared/'candidates.jsonl'); rows[0]['response']='Locally rehashed false response'
        self.f.write_rows(self.prepared/'candidates.jsonl',rows)
        manifest['files']['candidates.jsonl']['sha256']=train._hash_file(self.prepared/'candidates.jsonl')
        train._write(self.prepared/'manifest.json',manifest)
        REAL_VERIFY(self.prepared)
        with self.assertRaises(ValueError):acceptance.verify_bundle(self.prepared)

    def test_incomplete_scores_cannot_freeze(self):
        self.f.score(200)
        with self.assertRaises(ValueError):acceptance.freeze(self.f.pool,self.f.output,self.prepared)
        self.assertFalse(self.prepared.exists())

    def test_audit_is_blinded_deterministic_training_only_and_never_invents_labels(self):
        result=self.audit_pack()
        train_ids=[f'f{i}' for i in range(130,730)]
        expected=sorted(train_ids,key=lambda i:(hashlib.sha256(('tiny-aya-quality-audit-v1:'+i).encode()).hexdigest(),i))[:30]
        self.assertEqual(result['family_ids'],expected)
        rows=guard._rows(self.audit/'rows.jsonl')
        self.assertEqual(len(rows),90)
        self.assertEqual([r['lang'] for r in rows],['en','ko','zh']*30)
        self.assertEqual(set(rows[0]),{'id','family_id','lang','instruction','response'})
        self.assertEqual(rows[0]['instruction'],f"Greet {rows[0]['family_id'][1:]}.")
        self.assertEqual(rows[0]['response'],'Hello en.')
        template=quality._read(self.audit/'annotations.template.json')
        self.assertIsNone(template['pack_manifest_sha256'])
        self.assertTrue(all(a['quality'] is None and a['reason'] is None for a in template['annotations']))
        self.assertFalse((self.audit/'annotations.json').exists())
        with self.assertRaises((ValueError,FileNotFoundError)):acceptance.accept(self.prepared,self.screened,self.audit)
        self.assertFalse((self.screened/'quality-acceptance.json').exists())

    def test_uncertainty_is_separate_and_does_not_impose_usable_threshold(self):
        self.audit_pack(); self.annotate(labels={i:('ambiguous' if i%2 else 'unclear') for i in range(90)})
        report=acceptance.accept(self.prepared,self.screened,self.audit)
        self.assertEqual(report['status'],'accepted_initial')
        verified=acceptance.verify_acceptance(self.prepared,self.screened)
        sidecar=quality._read(self.screened/'quality-acceptance.json')
        self.assertEqual(sidecar['status'],'provisional_independent_audit_acceptance')
        self.assertEqual(sidecar['calibration_qualification'],'not_established')
        self.assertEqual(sidecar['human_review'],'pending')
        self.assertEqual(sidecar['acceptance_basis'],'independent_blinded_training_source_audit')
        self.assertEqual(verified['quality_acceptance_sha256'],train._hash_file(self.screened/'quality-acceptance.json'))
        self.assertEqual(set(verified['bindings']),{'quality_acceptance_sha256','quality_evidence_sha256'})
        self.assertEqual(quality._read(self.audit/'report.json')['per_language_counts']['en']['denominator'],30)
        before=(self.audit/'review-manifest.json').read_bytes()
        self.assertEqual(acceptance.accept(self.prepared,self.screened,self.audit)['status'],'accepted_initial')
        self.assertEqual((self.audit/'review-manifest.json').read_bytes(),before)
        shutil.rmtree(self.audit)
        acceptance.verify_acceptance(self.prepared,self.screened)

    def test_four_errors_fail_and_closed_review_cannot_be_rewritten(self):
        self.audit_pack(); self.annotate(labels={0:'incorrect',3:'incorrect',6:'incorrect',9:'incorrect'})
        self.assertEqual(acceptance.accept(self.prepared,self.screened,self.audit)['status'],'failed_threshold')
        self.assertFalse((self.screened/'quality-acceptance.json').exists())
        with self.assertRaises(ValueError):acceptance.freeze(self.f.pool,self.f.output,self.root/'bad-repair',repair_audit=self.audit)
        self.annotate(labels={0:'incorrect'})
        with self.assertRaises(ValueError):acceptance.accept(self.prepared,self.screened,self.audit)

    def test_quotes_labels_order_and_invented_authority_fail_before_closure(self):
        self.audit_pack(); self.annotate(labels={0:'incorrect'})
        original=quality._read(self.audit/'annotations.json')
        for mutation in ('quote','field','label','order','reviewer','authority','extra','binding','empty_evidence','schema'):
            with self.subTest(mutation=mutation):
                value=copy.deepcopy(original)
                if mutation=='quote':value['annotations'][0]['source_evidence'][0]['quote']='not in source'
                if mutation=='field':value['annotations'][0]['source_evidence'][0]['field']='gold'
                if mutation=='label':value['annotations'][0]['quality']='unknown'
                if mutation=='order':value['annotations'].reverse()
                if mutation=='reviewer':value['reviewer']['method']=''
                if mutation=='authority':value['human_review']='approved'
                if mutation=='extra':value['annotations'][0]['expected_answer']='private key'
                if mutation=='binding':value['pack_manifest_sha256']=None
                if mutation=='empty_evidence':value['annotations'][0]['source_evidence']=[]
                if mutation=='schema':value['schema_version']=True
                train._write(self.audit/'annotations.json',value)
                with self.assertRaises(ValueError):acceptance.accept(self.prepared,self.screened,self.audit)
                self.assertFalse((self.audit/'review-manifest.json').exists())

    def repair(self, bad=None):
        self.audit_pack(); self.annotate(labels=bad or {0:'incorrect',4:'incorrect',8:'incorrect'})
        result=acceptance.accept(self.prepared,self.screened,self.audit)
        self.assertEqual(result['status'],'repair_required')
        original_report=(self.audit/'report.json').read_bytes()
        repaired=self.root/'repaired'; screened=self.root/'results'/'tiny-aya'/'repaired-safety'; audit=self.root/'replacement-review'
        acceptance.freeze(self.f.pool,self.f.output,repaired,repair_audit=self.audit)
        guard.screen_candidates(repaired,screened)
        result=acceptance.freeze_audit(repaired,screened,audit)
        return repaired,screened,audit,result,original_report

    def test_single_repair_preserves_eval_and_reviews_exact_additions(self):
        prepared,screened,audit,result,original_report=self.repair()
        self.assertEqual(REAL_VERIFY(prepared)['candidate_families'],897)
        self.assertEqual(result['mode'],'replacements')
        self.assertEqual(result['family_ids'],['f730','f731','f732'])
        self.assertEqual(result['row_count'],9)
        self.assertEqual(guard._rows(screened/'eval_task.jsonl'),guard._rows(self.screened/'eval_task.jsonl'))
        self.annotate(audit,labels={0:'unclear'})
        with self.assertRaises(ValueError):acceptance.accept(prepared,screened,audit)
        manifest_only=self.root/'manifest-only';manifest_only.mkdir()
        shutil.copyfile(self.audit/'review-manifest.json',manifest_only/'review-manifest.json')
        with self.assertRaises(ValueError):acceptance.accept(prepared,screened,audit,previous=manifest_only)
        result=acceptance.accept(prepared,screened,audit,previous=self.audit)
        self.assertEqual(result['status'],'accepted_after_repair')
        acceptance.verify_acceptance(prepared,screened)
        self.assertEqual((self.audit/'report.json').read_bytes(),original_report)
        with self.assertRaises(ValueError):acceptance.freeze(self.f.pool,self.f.output,self.root/'second-repair',repair_audit=audit)

    def test_any_replacement_error_terminates_without_sidecar(self):
        prepared,screened,audit,_,_=self.repair({0:'incorrect'})
        self.annotate(audit,labels={1:'incorrect'})
        result=acceptance.accept(prepared,screened,audit,previous=self.audit)
        self.assertEqual(result['status'],'failed_replacement_review')
        self.assertFalse((screened/'quality-acceptance.json').exists())

    def test_incomplete_safety_refuses_audit(self):
        self.freeze(); guard.screen_candidates(self.prepared,self.screened,200)
        with self.assertRaises(ValueError):acceptance.freeze_audit(self.prepared,self.screened,self.audit)

    def test_evidence_copy_is_self_contained_after_original_sources_removed(self):
        self.freeze()
        for directory in (self.f.pool,self.f.output,self.f.prepared,self.f.calibration,self.f.review,self.f.ready.parent):
            shutil.rmtree(directory)
        self.assertEqual(acceptance.verify_bundle(self.prepared)['candidate_families'],900)
        with tempfile.TemporaryDirectory() as moved:
            destination=Path(moved)/'accepted-candidates'
            shutil.copytree(self.prepared,destination);shutil.rmtree(self.prepared)
            with patch.dict(os.environ,{'TINY_AYA_EVIDENCE_ROOT':moved}):
                self.assertEqual(acceptance.verify_bundle(destination)['candidate_families'],900)

    def test_path_and_evidence_inventory_tampering_fail(self):
        self.freeze()
        manifest=quality._read(self.prepared/'manifest.json')
        original=copy.deepcopy(manifest)
        manifest['quality_curation']['quality_evidence']['path']='/tmp/manifest.json'
        train._write(self.prepared/'manifest.json',manifest)
        with self.assertRaises(ValueError):acceptance.verify_bundle(self.prepared)
        train._write(self.prepared/'manifest.json',original)
        unbound=self.prepared/'unbound.json';unbound.write_text('{}')
        with self.assertRaises(ValueError):acceptance.verify_bundle(self.prepared)
        unbound.unlink()
        extra=self.prepared/'quality-evidence'/'unknown.json'; extra.write_text('{}')
        with self.assertRaises(ValueError):acceptance.verify_bundle(self.prepared)
        extra.unlink()
        candidates=self.prepared/'candidates.jsonl'; blob=candidates.read_bytes(); candidates.unlink()
        outside=self.root/'outside.jsonl';outside.write_bytes(blob);candidates.symlink_to(outside)
        with self.assertRaises(ValueError):acceptance.verify_bundle(self.prepared)

    def test_sidecar_rebinding_and_rehashed_flags_cannot_grant_acceptance(self):
        self.audit_pack();self.annotate();acceptance.accept(self.prepared,self.screened,self.audit)
        path=self.screened/'quality-acceptance.json';original=path.read_bytes();value=json.loads(original)
        value['human_review']='approved';train._write(path,value)
        with self.assertRaises(ValueError):acceptance.verify_acceptance(self.prepared,self.screened)
        path.write_bytes(original)
        other=self.root/'other-safety';shutil.copytree(self.screened,other)
        guard._write(other/'report.json',dict(status='invented'))
        with self.assertRaises(ValueError):acceptance.verify_acceptance(self.prepared,other)

    def test_cli_audit_returns_serializable_runtime_paths(self):
        import contextlib
        import io
        self.freeze();guard.screen_candidates(self.prepared,self.screened)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            acceptance.main(['freeze-audit','--prepared',str(self.prepared),'--screened',str(self.screened),'--output',str(self.audit)])
        self.assertEqual(json.loads(output.getvalue())['status'],'awaiting_independent_review')

    def test_prepared_directory_symlink_cannot_bypass_path_policy(self):
        self.freeze()
        alias=self.root/'alias';alias.symlink_to(self.prepared,target_is_directory=True)
        with self.assertRaises(ValueError):acceptance.verify_bundle(alias)

    def test_three_incorrect_each_language_can_remove_nine_distinct_families(self):
        prepared,_,_,result,_=self.repair({i:'incorrect' for i in (0,3,6,10,13,16,20,23,26)})
        self.assertEqual(REAL_VERIFY(prepared)['candidate_families'],891)
        self.assertEqual(result['row_count'],27)
        self.assertEqual(result['family_ids'],[f'f{i}' for i in range(730,739)])

    def test_changed_retained_safety_result_cannot_expand_replacement_set(self):
        prepared,_,_,_,_=self.repair({0:'incorrect'})
        bad=quality._read(self.audit/'report.json')['confirmed_bad_family_ids']
        kept=next(f'f{i}' for i in range(130,730) if f'f{i}' not in bad)
        self.safety.fail=(kept+':en','prompt')
        another=self.root/'results'/'tiny-aya'/'changed-safety'
        guard.screen_candidates(prepared,another)
        with self.assertRaises(ValueError):acceptance.freeze_audit(prepared,another,self.root/'extra-review')

    def test_existing_sidecar_cannot_be_reused_for_a_different_completed_review(self):
        self.audit_pack();self.annotate();acceptance.accept(self.prepared,self.screened,self.audit)
        another=self.root/'other-audit'
        acceptance.freeze_audit(self.prepared,self.screened,another);self.annotate(another)
        value=quality._read(another/'annotations.json');value['reviewer']['name']='Different review process'
        train._write(another/'annotations.json',value)
        with self.assertRaises(ValueError):acceptance.accept(self.prepared,self.screened,another)
