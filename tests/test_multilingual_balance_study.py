"""Synthetic-only tests for training pool provenance and matched exposure."""
import copy
import importlib
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from experiments.multilingual_repair import train as old


class SyntheticTokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        prompt = [1, 7, 8]
        return prompt if add_generation_prompt else prompt + [11] * len(messages[-1]['content']) + [2]
    def __call__(self, text, *, add_special_tokens):
        return {'input_ids':[11]*len(text)}
    def decode(self, tokens, **kwargs):
        return 'x' * len(tokens)


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('experiments.language_balance.study')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prepared = self.root / 'prepared'; self.prepared.mkdir()
        config = json.loads((old.ROOT / 'experiments/multilingual_repair/config.json').read_bytes())
        old.write_json(self.prepared / 'config.json', config)
        self.rows = []
        for split, counts in [('train', (450,100,450)), ('calibration',(6,6,6)), ('alignment',(24,24,24))]:
            rows = []
            for language, count in zip(('en','ko','zh'), counts):
                for i in range(count):
                    row = dict(id=f'{split}-{language}-{i}', source_id=f'public-{language}-{i}', split=split,
                               language=language, request='synthetic question', system='synthetic system',
                               input_ids=[1,7,8], attention_mask=[1,1,1], prompt_tokens=3, sequence_tokens=3)
                    if split != 'alignment':
                        row.update(reference='x', category='task', input_ids=[1,7,8,11,2], attention_mask=[1]*5,
                                   labels=[-100]*3+[11,2], sequence_tokens=5, supervised_tokens=2)
                    rows.append(row)
            (self.prepared / (split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            if split == 'train': self.rows = rows
        # These opaque bytes must never be parsed by training preparation.
        for name in ('development.jsonl','confirmation.jsonl'):
            (self.prepared / name).write_bytes(b'opaque holdout bytes, not JSON\n')
        manifest = dict(status='passed_tokenization_preflight', truncated_examples=0,
                        tokenizer=dict(actual_snapshot_commit=old.REVISION,eos_token_id=2),
                        files={p.name:old.sha_bytes(p.read_bytes()) for p in self.prepared.iterdir()})
        old.write_json(self.prepared / 'manifest.json',manifest)
        self.audits=[]
        for label in ('a','b'):
            folder=self.root / ('audit-'+label);folder.mkdir();self.audits.append(folder)
            annotations=[dict(id=r['id'],quality='usable' if r['language']!='ko' or int(r['id'].rsplit('-',1)[1])<34 else ('incorrect' if int(r['id'].rsplit('-',1)[1])<52 else 'unclear'),
                              safety='benign',notes='synthetic evidence') for r in self.rows if r['language']=='ko']
            old.write_json(folder/'annotations.json',annotations)
            old.write_json(folder/'report.json',dict(**({'audit_type':'provisional_automated_training_data_audit'} if label=='a' else {'audit_kind':'independent provisional Korean training-data audit','review_status':'model-generated labels; not human certification'}),source_train_sha256=old.sha_bytes((self.prepared/'train.jsonl').read_bytes()),annotations_sha256=old.sha_bytes((folder/'annotations.json').read_bytes())))
        self.repairs=[dict(id=f'train-ko-{i}',status='repaired' if i==34 else 'excluded',reference='xx' if i==34 else '',
                           rationale='synthetic correction',sources=[]) for i in range(34,52)]
        self.review=[dict(id='train-ko-34',quality='usable',safety='benign',notes='synthetic independent assessment')]
        self.save_sources()

    def save_sources(self):
        repairdir=self.root/'repairs';repairdir.mkdir(exist_ok=True)
        gate={r['id']:r for r in self.rows if r['language']=='ko' and 34<=int(r['id'].rsplit('-',1)[1])<52}
        inputrows=[{**{k:r[k] for k in ('id','source_id','language','request','system','reference')},'audit_notes':{}} for r in gate.values()]
        (repairdir/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputrows))
        old.write_json(repairdir/'annotations.json',self.repairs)
        sourcehash=old.sha_bytes((self.prepared/'train.jsonl').read_bytes())
        old.write_json(repairdir/'report.json',dict(status='completed_provisional_target_repair',records=18,
            inputs_sha256=old.sha_bytes((repairdir/'inputs.jsonl').read_bytes()),annotations_sha256=old.sha_bytes((repairdir/'annotations.json').read_bytes()),source_train_sha256=sourcehash))
        pack=[dict(id=r['id'],request='synthetic question',system='synthetic system',reference=r['reference'],sources=r['sources']) for r in self.repairs if r['status']=='repaired']
        for name in ('repair-a','repair-b'):
            folder=self.root/name;folder.mkdir(exist_ok=True)
            (folder/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pack))
            old.write_json(folder/'annotations.json',self.review)
            old.write_json(folder/'report.json',dict(status='completed_provisional_target_review',records=len(self.review),
                inputs_sha256=old.sha_bytes((folder/'inputs.jsonl').read_bytes()),annotations_sha256=old.sha_bytes((folder/'annotations.json').read_bytes()),
                repair_proposals_sha256=old.sha_bytes((repairdir/'annotations.json').read_bytes()),source_train_sha256=sourcehash))

    def prepare(self):
        return self.api.prepare_study(prepared=self.prepared,audit_a=self.audits[0],audit_b=self.audits[1],
            repairs=self.root/'repairs',repair_review_a=self.root/'repair-a',repair_review_b=self.root/'repair-b',
            output=self.root/'study',tokenizer=SyntheticTokenizer())

    def test_pool_consensus_and_exact_exposure_without_holdout_parsing(self):
        self.prepare(); value=self.api.load_study(self.root/'study',self.prepared)
        self.assertEqual(Counter(r['language'] for r in value['pool']), {'en':450,'ko':35,'zh':450})
        self.assertEqual(value['pool_by_id']['train-ko-34']['reference'],'xx')
        self.assertEqual(value['pool_by_id']['train-ko-34']['input_ids'],[1,7,8,11,11,2])
        self.assertNotIn('train-ko-35',value['pool_by_id'])
        for arm,counts in [('control',{'en':450,'ko':100,'zh':450}),('balanced',{'en':333,'ko':334,'zh':333})]:
            self.assertEqual(len(value['schedules'][arm]),1000)
            self.assertEqual(Counter(value['pool_by_id'][i]['language'] for i in value['schedules'][arm]),counts)
        ko=lambda arm:[i for i in value['schedules'][arm] if value['pool_by_id'][i]['language']=='ko']
        self.assertEqual(ko('control'),ko('balanced')[:100])
        self.assertEqual(len(set(ko('balanced')[:35])),35)
        self.assertEqual(len(value['alignment_order']),1000)
        self.assertEqual(value['bundle']['blobs']['confirmation.jsonl'],b'opaque holdout bytes, not JSON\n')

    def test_review_disagreement_excludes_repair_and_preserves_originals(self):
        review=copy.deepcopy(self.review);review[0]['safety']='unclear'
        old.write_json(self.root/'repair-b/annotations.json',review)
        report=json.loads((self.root/'repair-b/report.json').read_bytes());report['annotations_sha256']=old.sha_bytes((self.root/'repair-b/annotations.json').read_bytes());old.write_json(self.root/'repair-b/report.json',report)
        self.prepare();value=self.api.load_study(self.root/'study',self.prepared)
        self.assertEqual(Counter(r['language'] for r in value['pool'])['ko'],34)
        self.assertEqual(value['pool_by_id']['train-en-0'],self.rows[0])

    def test_rejects_unapproved_repair_identity_and_duplicate_review(self):
        for mutation in ('wrong_id','duplicate'):
            with self.subTest(mutation=mutation):
                self.save_sources()
                if mutation=='wrong_id':
                    repairs=copy.deepcopy(self.repairs);repairs[0]['id']='calibration-ko-0';old.write_json(self.root/'repairs/annotations.json',repairs)
                else:old.write_json(self.root/'repair-a/annotations.json',self.review*2)
                with self.assertRaises(ValueError):self.prepare()

    def test_target_cap_and_changed_prompt_are_rejected_without_truncation(self):
        self.repairs[0]['reference']='x'*193;self.save_sources()
        with self.assertRaisesRegex(ValueError,'192|target'):self.prepare()
        self.repairs[0]['reference']='xx';self.save_sources()
        tokenizer=SyntheticTokenizer()
        tokenizer.apply_chat_template=lambda *a,**k:[99,7,8,11,11,2]
        with self.assertRaisesRegex(ValueError,'prompt'):
            self.api.prepare_study(prepared=self.prepared,audit_a=self.audits[0],audit_b=self.audits[1],
                repairs=self.root/'repairs',repair_review_a=self.root/'repair-a',repair_review_b=self.root/'repair-b',
                output=self.root/'study',tokenizer=tokenizer)

    def test_tampering_and_self_consistent_wrong_schedule_are_rejected(self):
        self.prepare(); folder=self.root/'study'
        path=folder/'inputs.jsonl'; original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'hash'):self.api.load_study(folder,self.prepared)
        path.write_bytes(original)
        config=json.loads((folder/'training-config.json').read_bytes());config['schedules']['balanced'][0]='calibration-en-0'
        old.write_json(folder/'training-config.json',config)
        manifest=json.loads((folder/'manifest.json').read_bytes());manifest['files']['training-config.json']=old.sha_bytes((folder/'training-config.json').read_bytes())
        old.write_json(folder/'manifest.json',manifest)
        with self.assertRaises(ValueError):self.api.load_study(folder,self.prepared)

    def reseal(self, name):
        folder=self.root/'study';manifest=json.loads((folder/'manifest.json').read_bytes())
        manifest['files'][name]=old.sha_bytes((folder/name).read_bytes());old.write_json(folder/'manifest.json',manifest)

    def test_192_raw_target_tokens_plus_EOS_are_accepted(self):
        self.repairs[0]['reference']='x'*192;self.save_sources()
        value=self.prepare()
        self.assertEqual(value['pool_by_id']['train-ko-34']['supervised_tokens'],193)

    def test_original_pool_retention_does_not_tighten_raw_token_cap_to_suffix_length(self):
        path=self.prepared/'train.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0].update(reference='x'*192,input_ids=[1,7,8]+[11]*193+[2],attention_mask=[1]*197,
                       labels=[-100]*3+[11]*193+[2],sequence_tokens=197,supervised_tokens=194)
        self.rows=rows;path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=json.loads((self.prepared/'manifest.json').read_bytes());manifest['files']['train.jsonl']=old.sha_bytes(path.read_bytes());old.write_json(self.prepared/'manifest.json',manifest)
        for folder in self.audits:
            report=json.loads((folder/'report.json').read_bytes());report['source_train_sha256']=old.sha_bytes(path.read_bytes());old.write_json(folder/'report.json',report)
        self.save_sources();value=self.prepare()
        self.assertEqual(value['pool_by_id']['train-en-0']['supervised_tokens'],194)

    def test_self_consistent_stale_repair_tokens_are_rejected(self):
        self.prepare();path=self.root/'study/inputs.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        row=next(r for r in rows if r['id']=='train-ko-34');row['input_ids'][-2]=12;row['labels'][-2]=12
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows));self.reseal('inputs.jsonl')
        with self.assertRaisesRegex(ValueError,'token|annotations'):self.api.load_study(self.root/'study',self.prepared)

    def test_original_audit_source_hash_and_same_reviewer_directory_rejected(self):
        report=json.loads((self.audits[0]/'report.json').read_bytes());report['source_train_sha256']='0'*64
        old.write_json(self.audits[0]/'report.json',report)
        with self.assertRaisesRegex(ValueError,'audit'):self.prepare()
        report['source_train_sha256']=old.sha_bytes((self.prepared/'train.jsonl').read_bytes());old.write_json(self.audits[0]/'report.json',report)
        with self.assertRaisesRegex(ValueError,'distinct'):
            self.api.prepare_study(prepared=self.prepared,audit_a=self.audits[0],audit_b=self.audits[1],
                repairs=self.root/'repairs',repair_review_a=self.root/'repair-a',repair_review_b=self.root/'repair-a',
                output=self.root/'study',tokenizer=SyntheticTokenizer())

    def test_changed_source_bytes_rejected_after_preparation(self):
        self.prepare();folder=self.root/'study'
        annotations=json.loads((folder/'annotations.json').read_bytes());annotations['sources']['repairs-annotations']['utf8']='[]'
        old.write_json(folder/'annotations.json',annotations)
        manifest=json.loads((folder/'manifest.json').read_bytes());manifest['files']['annotations.json']=old.sha_bytes((folder/'annotations.json').read_bytes());old.write_json(folder/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'source.*hash|provenance'):self.api.load_study(self.root/'study',self.prepared)


if __name__=='__main__':unittest.main()
