"""CPU evidence tests for model-independent, left-padded local generation."""
import copy
import unittest

import torch
from transformers import GenerationConfig

from experiments.tiny_aya import runtime


class Tokenizer:
    def __len__(self): return 32
    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is True and clean_up_tokenization_spaces is False
        return ''.join(chr(96+i) for i in ids if i not in (0,3,6))


class Model:
    def __init__(self, suffixes, *, wrong_prefix=False, nonfinite=False):
        self.suffixes=suffixes; self.wrong_prefix=wrong_prefix;self.nonfinite=nonfinite
        self.training=False
    def generate(self, *, input_ids, attention_mask, generation_config, logits_processor):
        assert not torch.is_grad_enabled()
        self.inputs=input_ids.tolist();self.masks=attention_mask.tolist()
        scores=torch.zeros((len(self.suffixes),32))
        if self.nonfinite:scores[0,0]=float('nan')
        for processor in logits_processor:processor(input_ids,scores)
        prefix=input_ids.clone()
        if self.wrong_prefix:prefix[0,0]=9
        return torch.cat((prefix,torch.tensor(self.suffixes)),dim=1)


def config(*, eos=(3,6),pad=0,cap=4):
    return GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=cap,eos_token_id=list(eos),pad_token_id=pad,num_return_sequences=1,return_dict_in_generate=False)


class RuntimeTests(unittest.TestCase):
    def test_resolved_configuration_is_generic_and_independent(self):
        one=runtime.resolved_generation_config(max_new_tokens=3,eos_token_id=[11],pad_token_id=0,bos_token_id=5)
        two=runtime.resolved_generation_config(max_new_tokens=4,eos_token_id=[6],pad_token_id=6,bos_token_id=5)
        self.assertEqual(one.eos_token_id,[11]);self.assertEqual(two.pad_token_id,6)
        self.assertFalse(one.do_sample);self.assertEqual(one.num_beams,1)
        self.assertIsNone(one.suppress_tokens)
        one.eos_token_id.append(12)
        self.assertEqual(two.eos_token_id,[6])
        result=runtime.generate_batch(Model([[7,11,0],[8,9,11]]),Tokenizer(),[[5],[5,4]],generation_config=one,device='cpu')
        self.assertEqual(result[0]['generated_ids'],[7,11])
        self.assertEqual(result[0]['response'],'gk')

    def test_left_mask_padded_prefix_and_early_eos_with_long_sibling(self):
        model=Model([[7,6,0,0],[8,9,10,3]])
        rows=runtime.generate_batch(model,Tokenizer(),[[1,4],[1,4,5,7]],generation_config=config(),device='cpu')
        self.assertEqual(model.inputs,[[0,0,1,4],[1,4,5,7]])
        self.assertEqual(model.masks,[[0,0,1,1],[1,1,1,1]])
        self.assertEqual(rows[0]['raw_generated_ids'],[7,6,0,0])
        self.assertEqual(rows[0]['generated_ids'],[7,6])
        self.assertEqual(rows[0]['response'],'g')
        self.assertEqual(rows[0]['post_eos_pad_count'],2)
        self.assertEqual(rows[0]['pad_count'],2)
        self.assertEqual(rows[0]['eos_index'],1)
        self.assertEqual(rows[1]['finish_reason'],'eos')
        self.assertEqual(rows[1]['completion_tokens'],4)
        self.assertEqual(rows[0]['padded_width'],4)
        self.assertEqual(rows[0]['input_ids'],[0,0,1,4])
        self.assertEqual(rows[0]['attention_mask'],[0,0,1,1])
        self.assertFalse(model.training)

    def test_shared_pad_eos_is_not_deleted_and_real_pad_prompt_has_mask_one(self):
        model=Model([[6,6,6],[7,8,6]])
        rows=runtime.generate_batch(model,Tokenizer(),[[1,6],[1,4,6]],generation_config=config(eos=(6,),pad=6,cap=3),device='cpu')
        self.assertEqual(model.masks,[[0,1,1],[1,1,1]])
        self.assertEqual(rows[0]['generated_ids'],[6])
        self.assertEqual(rows[0]['post_eos_pad_count'],2)
        self.assertTrue(rows[0]['empty_response'])
        self.assertEqual(rows[0]['finish_reason'],'eos')

    def test_eos_on_cap_complete_but_no_eos_capped_and_short_without_eos_rejected(self):
        rows=runtime.generate_batch(Model([[7,8,9,6],[7,8,9,10]]),Tokenizer(),[[1],[1]],generation_config=config(),device='cpu')
        self.assertEqual([r['finish_reason'] for r in rows],['eos','length'])
        self.assertEqual(rows[1]['eos_id'],None)
        with self.assertRaises(ValueError):runtime.generate_batch(Model([[7,8]]),Tokenizer(),[[1]],generation_config=config(),device='cpu')

    def test_token_512_eos_is_complete_and_no_eos_is_length(self):
        at_cap=runtime.completion_evidence(Tokenizer(),[1],[7]*511+[6],generation_config=config(cap=512),padded_width=1,elapsed_batch_seconds=0.)
        capped=runtime.completion_evidence(Tokenizer(),[1],[7]*512,generation_config=config(cap=512),padded_width=1,elapsed_batch_seconds=0.)
        self.assertEqual(at_cap['eos_index'],511)
        self.assertEqual(at_cap['completion_tokens'],512)
        self.assertEqual(at_cap['finish_reason'],'eos')
        self.assertEqual(capped['finish_reason'],'length')

    def test_rejects_bad_prefix_nonpad_trailer_nonfinite_and_invalid_tokens(self):
        for model in (Model([[7,6,0,0]],wrong_prefix=True),Model([[7,6,9,0]])):
            with self.assertRaises(ValueError):runtime.generate_batch(model,Tokenizer(),[[1]],generation_config=config(),device='cpu')
        with self.assertRaises(FloatingPointError):runtime.generate_batch(Model([[7,6,0,0]],nonfinite=True),Tokenizer(),[[1]],generation_config=config(),device='cpu')
        for ids in ([],[[]],[[True]],[[1.0]],[[-1]],[[32]]):
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                runtime.generate_batch(Model([[6]]),Tokenizer(),ids,generation_config=config(),device='cpu')
        bad_returns=config();bad_returns.num_return_sequences=True
        for bad in (config(pad=True),config(eos=(True,)),config(cap=True),bad_returns):
            with self.assertRaises(ValueError):runtime.generate_batch(Model([[6]]),Tokenizer(),[[1]],generation_config=bad,device='cpu')

    def test_real_transformers_checkpoint_suppressions_are_exposed(self):
        from transformers import Cohere2Config,Cohere2ForCausalLM
        from experiments.tiny_aya.evaluate import generation_config
        previous=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,previous)
        model=Cohere2ForCausalLM(Cohere2Config(vocab_size=32,hidden_size=16,intermediate_size=24,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32,
            sliding_window=16,attention_dropout=0.,bos_token_id=5,eos_token_id=3,pad_token_id=0)).eval()
        frozen=generation_config();frozen.max_new_tokens=4
        model.generation_config.suppress_tokens=[7]
        with self.assertRaisesRegex(ValueError,'effective generation'):
            runtime.generate_batch(model,Tokenizer(),[[5,4]],generation_config=frozen,device='cpu')
        self.assertEqual(model.generation_config.suppress_tokens,[7])
        model.generation_config=copy.deepcopy(frozen)
        with self.assertLogs('transformers.generation.utils',level='WARNING') as logged:
            rows=runtime.generate_batch(model,Tokenizer(),[[5,4],[5,4,8]],generation_config=frozen,device='cpu')
        self.assertEqual(len(logged.records),1)
        self.assertIn('Both `max_new_tokens` (=4) and `max_length`(=20)',logged.records[0].getMessage())
        self.assertIn('`max_new_tokens` will take precedence',logged.records[0].getMessage())
        self.assertEqual(len(rows),2)
        self.assertEqual(model.generation_config.to_dict(),frozen.to_dict())

    def test_reconstruct_exact_metadata_and_reject_inconsistent_evidence(self):
        row=runtime.generate_batch(Model([[7,6,0,0]]),Tokenizer(),[[1]],generation_config=config(),device='cpu')[0]
        runtime.validate_evidence(row,Tokenizer(),[1],generation_config=config(),padded_width=1)
        for key,value in [('generated_ids',[7]),('response','changed'),('eos_index',2),('attention_mask',[0]),('completion_tokens',4),('elapsed_batch_seconds',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                runtime.validate_evidence({**copy.deepcopy(row),key:value},Tokenizer(),[1],generation_config=config(),padded_width=1)

if __name__=='__main__':unittest.main()
