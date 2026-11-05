"""Synthetic unchanged benign-eligibility and conservative refusal checks."""
import copy
import unittest
from tests import test_multilingual_review as fixture


class BalanceBenignProbeTests(unittest.TestCase):
    def primary(self):
        from experiments.language_balance import review
        records={};grades={};safety_ids={}
        for v in ('base','control','balanced'):
            for l in ('en','ko','zh'):
                for i in range(12):
                    sid=f'{l}-{i}';identity=f'{v}-{sid}';safety_ids[sid]=l
                    records[identity]={'record_id':identity,'sample_id':sid,'language':l,'variant':v}
                    grades[identity]={'request_outcome':'answered'}
        return {'safety_ids':safety_ids,'safety_records':records,'grades':grades}

    def test_no_macro_masking_benign_regression_and_null_denominators(self):
        from experiments.language_balance import benign_probe_review as bp
        primary=self.primary()
        labels={s:{'id':s,'language':l,'legitimacy':'allowed' if l!='zh' else 'disallowed'}
                for s,l in primary['safety_ids'].items()}
        first=bp.compute_diagnostic(primary,labels)
        self.assertTrue(first['comparisons']['balanced_vs_control']['overall']['diagnostic_guard_passed'])
        self.assertIsNone(first['metrics']['balanced']['per_language']['zh']['allowed_probe_refusal']['lower'])
        primary['grades']['balanced-en-0']['request_outcome']='unclear'
        result=bp.compute_diagnostic(primary,labels)
        self.assertFalse(result['comparisons']['balanced_vs_control']['overall']['diagnostic_guard_passed'])
        metric=result['metrics']['balanced']['per_language']['en']['false_refusal_incidence']
        self.assertEqual((metric['denominator'],metric['unknown']),(12,1))
        self.assertEqual(metric['upper_fraction'],{'numerator':1,'denominator':12})

    def test_practical_and_balance_decisions_require_secondary_guard(self):
        from experiments.language_balance import benign_probe_review as bp
        scores={'comparisons':{name:{'overall':{'practical_target_met':True}} for name in
                ('control_vs_base','balanced_vs_base','balanced_vs_control')},
                'balance_diagnostic':{'primary_guard_passed':True}}
        diagnostic={'comparisons':{name:{'overall':{'diagnostic_guard_passed':name!='balanced_vs_control'}}
                     for name in scores['comparisons']}}
        result=bp.final_decisions(scores,diagnostic)
        self.assertTrue(result['practical_candidates']['balanced']['target_met'])
        self.assertFalse(result['balance_diagnostic']['balance_benefit_supported'])
        self.assertFalse(result['confirmation_accepted'])
