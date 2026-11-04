"""Model-independent left-padded generation and independently checkable evidence.

The first EOS is retained even when EOS equals PAD. Only the validated trailer
following it is padding. No model/tokenizer globals are modified here.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time


def _require(condition, message):
    if not condition: raise ValueError(message)


def token_ids(ids, vocabulary, *, empty=False):
    _require(type(ids) is list and (empty or bool(ids)) and
             all(type(i) is int and 0 <= i < vocabulary for i in ids), 'Invalid token IDs')
    return ids


def resolved_generation_config(**overrides):
    """Fresh greedy settings with global defaults, independent of any checkpoint.

    EOS/PAD/BOS and token limits belong to the caller. The caller must install
    this same baseline on its independently owned model before generation;
    generate_batch verifies the effective configuration without mutating it.
    """
    from transformers import GenerationConfig
    defaults=GenerationConfig._get_default_generation_params()
    defaults.update(do_sample=False,num_beams=1,num_return_sequences=1,use_cache=True,
        return_dict_in_generate=False,output_scores=False,output_logits=False,disable_compile=True)
    defaults.update(copy.deepcopy(overrides))
    return GenerationConfig(**defaults)


def generation_policy(config, tokenizer):
    from transformers import GenerationConfig
    _require(isinstance(config, GenerationConfig), 'Expected explicit GenerationConfig')
    eos=config.eos_token_id
    eos=[eos] if type(eos) is int else eos
    token_ids(eos,len(tokenizer))
    _require(len(set(eos))==len(eos), 'Duplicate EOS IDs')
    token_ids([config.pad_token_id],len(tokenizer))
    _require(type(config.max_new_tokens) is int and config.max_new_tokens>0, 'Invalid generation cap')
    _require(config.do_sample is False and type(config.num_beams) is int and config.num_beams==1 and
             type(config.num_return_sequences) is int and config.num_return_sequences==1 and
             config.return_dict_in_generate is False,
             'Only greedy single-sequence generation is supported')
    return eos,config.pad_token_id,config.max_new_tokens


def completion_evidence(tokenizer, input_ids, raw_generated_ids, *, generation_config,
                        padded_width, elapsed_batch_seconds):
    """Reconstruct a row from raw suffix tokens, rejecting impossible stop evidence."""
    eos,pad,cap=generation_policy(generation_config,tokenizer)
    token_ids(input_ids,len(tokenizer))
    token_ids(raw_generated_ids,len(tokenizer),empty=True)
    _require(type(padded_width) is int and padded_width>=len(input_ids), 'Invalid padded width')
    _require(type(elapsed_batch_seconds) in (int,float) and math.isfinite(elapsed_batch_seconds)
             and elapsed_batch_seconds>=0, 'Invalid batch duration')
    _require(0<len(raw_generated_ids)<=cap, 'Invalid generated suffix length')
    first=next((i for i,t in enumerate(raw_generated_ids) if t in eos),None)
    if first is None:
        _require(len(raw_generated_ids)==cap, 'Non-EOS completion ended before cap')
        generated=raw_generated_ids[:];trailer=[]
    else:
        generated=raw_generated_ids[:first+1];trailer=raw_generated_ids[first+1:]
        _require(all(t==pad for t in trailer), 'Non-padding token after first EOS')
    response=tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    _require(type(response) is str, 'Decoded response must be text')
    count=padded_width-len(input_ids)
    return dict(raw_generated_ids=raw_generated_ids[:],generated_ids=generated,response=response,
        response_sha256=hashlib.sha256(response.encode('utf-8')).hexdigest(),
        finish_reason='eos' if first is not None else 'length',prompt_tokens=len(input_ids),
        padded_width=padded_width,pad_count=count,post_eos_pad_count=len(trailer),
        eos_id=generated[-1] if first is not None else None,eos_index=first,
        completion_tokens=len(generated),empty_response=not bool(response.strip()),
        input_ids=[pad]*count+input_ids,attention_mask=[0]*count+[1]*len(input_ids),
        elapsed_batch_seconds=elapsed_batch_seconds)


def validate_evidence(row, tokenizer, input_ids, *, generation_config, padded_width):
    """Validate saved metadata by re-parsing tokens and re-decoding text on CPU."""
    _require(isinstance(row,dict),'Invalid generation evidence')
    rebuilt=completion_evidence(tokenizer,input_ids,row.get('raw_generated_ids'),
        generation_config=generation_config,padded_width=padded_width,
        elapsed_batch_seconds=row.get('elapsed_batch_seconds'))
    for key,value in rebuilt.items():
        # JSON equality alone accepts bool/int or int/float substitutions.
        _require(type(row.get(key)) is type(value) and row[key]==value, f'Inconsistent generation evidence: {key}')
    for key in ('generated_ids','input_ids','attention_mask'):
        _require(all(type(i) is int for i in row[key]), f'Invalid integer evidence: {key}')
    return rebuilt


class FiniteLogits:
    """Raise on nonfinite scores; never sanitize or alter the scores."""
    def __call__(self, input_ids, scores):
        import torch
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError('Non-finite generation logits')
        return scores


def generate_batch(model, tokenizer, input_ids, *, generation_config, device):
    """Generate one frozen batch with exact left masks and padded-prefix checks."""
    import torch
    from transformers import LogitsProcessorList
    _,pad,_=generation_policy(generation_config,tokenizer)
    _require(type(input_ids) is list and bool(input_ids),'Empty/invalid batch')
    for ids in input_ids:token_ids(ids,len(tokenizer))
    _require(getattr(model,'training',None) is False,'Model must be in eval mode')
    if hasattr(model,'_prepare_generation_config'):
        effective,kwargs=model._prepare_generation_config(copy.deepcopy(generation_config))
        _require(effective.to_dict()==generation_config.to_dict() and not kwargs,
                 'Unfrozen effective generation configuration')
    width=max(map(len,input_ids))
    padded=[[pad]*(width-len(ids))+ids for ids in input_ids]
    masks=[[0]*(width-len(ids))+[1]*len(ids) for ids in input_ids]
    ids_tensor=torch.tensor(padded,dtype=torch.long,device=device)
    mask_tensor=torch.tensor(masks,dtype=torch.long,device=device)
    if str(device)=='mps':torch.mps.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(input_ids=ids_tensor,attention_mask=mask_tensor,
            generation_config=copy.deepcopy(generation_config),logits_processor=LogitsProcessorList([FiniteLogits()]))
    if str(device)=='mps':torch.mps.synchronize()
    elapsed=time.perf_counter()-start
    _require(isinstance(output,torch.Tensor) and output.ndim==2 and output.shape[0]==len(input_ids)
             and output.shape[1]>width and output.dtype in (torch.int64,torch.int32), 'Invalid generated tensor')
    output=output.detach().cpu().tolist()
    _require(all(row[:width]==prefix for row,prefix in zip(output,padded)),'Returned padded input prefix mismatch')
    return [completion_evidence(tokenizer,ids,row[width:],generation_config=generation_config,
                padded_width=width,elapsed_batch_seconds=elapsed) for ids,row in zip(input_ids,output)]
