from vllm import LLM, SamplingParams
from typing import List, Dict
from transformers import PreTrainedTokenizer
from openai import OpenAI
from tqdm import tqdm
import time
def generate_vanilla(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    Generate with no budget forcing.
    """
    prompt = [tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False) for message in messages]
    outputs = llm.generate(prompt, sampling_params)
    for i in range(len(outputs)):
        outputs[i] = (outputs[i].outputs[0].text, len(outputs[i].outputs[0].token_ids))
    return outputs

def generate_prompted(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict],sampling_params: SamplingParams) -> str:
    """
    Add "Think in less xxx tokens" after user prompt.
    """
    for message in messages:
        for m in message:
            if m["role"] == "user":
                m["content"] += f" Think in less than {sampling_params.max_tokens - 30} tokens."
    prompt = [tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False) for message in messages]
    outputs = llm.generate(prompt, sampling_params)
    for i in range(len(outputs)):
        outputs[i] = (outputs[i].outputs[0].text, len(outputs[i].outputs[0].token_ids))
    return outputs


def generate_s1(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    s1 style budget forcing: after "budget" token is used, if thinking does not finish, add stop thinking token and output answer.
    """
    prompt = [tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False) for message in messages]
    outputs = llm.generate(prompt, sampling_params)
    prompt2 = []
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            prompt2.append(output.outputs[0].text + "\n</think>\nThe final answer is: \boxed")
    sampling_params = SamplingParams(max_tokens=20, min_tokens = 0, temperature=0.0, stop_token_ids = [tokenizer.special_tokens_map["eos_token"]])
    outputs2 = llm.generate(prompt2, sampling_params)
    index = 0
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            output.outputs[0].text += "\n</think>\nThe final answer is: \boxed" + outputs2[index].outputs[0].text
            output.outputs[0].token_ids += outputs2[index].outputs[0].token_ids
            index += 1
    for i in range(len(outputs)):
        outputs[i] = (outputs[i].outputs[0].text, len(outputs[i].outputs[0].token_ids))
    return outputs

def generate_prompted_s1(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    Add "Think in less xxx tokens" after user prompt. 
    After "budget" token is used, if thinking does not finish, add stop thinking token and output answer.
    """
    for message in messages:
        for m in message:
            if m["role"] == "user":
                m["content"] += f" Think in less than {sampling_params.max_tokens - 30} tokens."
    prompt = [tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False) for message in messages]
    outputs = llm.generate(prompt, sampling_params)
    prompt2 = []
    # s1 forcing
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            prompt2.append(output.outputs[0].text + "\n</think>\nThe final answer is: \boxed")
    sampling_params = SamplingParams(max_tokens=20, min_tokens = 0, temperature=0.0, stop_token_ids = [tokenizer.special_tokens_map["eos_token"]])
    outputs2 = llm.generate(prompt2, sampling_params)
    index = 0
    # concatenate the outputs
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            output.outputs[0].text += "\n</think>\nThe final answer is: \boxed" + outputs2[index].outputs[0].text
            output.outputs[0].token_ids += outputs2[index].outputs[0].token_ids
            index += 1
    for i in range(len(outputs)):
        outputs[i] = (outputs[i].outputs[0].text, len(outputs[i].outputs[0].token_ids))
    return outputs


# openai generation

def generate_vanilla_openai(llm: OpenAI, model: str, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    Generate with no budget forcing.
    """
    while True:
        try:
            #print(messages)
            response = llm.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
            )
            #print(response)
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content != None:
                return dict(text='<think>' + response.choices[0].message.reasoning_content + "\n</think>\n" + response.choices[0].message.content, token_num=response.usage.completion_tokens)
            else:
                return dict(text=response.choices[0].message.content, token_num=response.usage.completion_tokens)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

def generate_prompted_s1_openai(llm: OpenAI, model: str, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    Add "Think in less xxx tokens" after user prompt. 
    After "budget" token is used, if thinking does not finish, add stop thinking token and output answer.
    """
    for m in messages:
        if m["role"] == "user":
            m["content"] += f" Think in less than {sampling_params.max_tokens - 30} tokens."
    while True:
        try:
            response = llm.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
            )
            assert not hasattr(response.choices[0].message, 'reasoning_content') or response.choices[0].message.reasoning_content == None, "Remote API does not support s1 forcing"
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    if "</think>" not in response.choices[0].message.content:
        response.choices[0].message.content += "\n</think>\nThe final answer is: \boxed"
    while True:
        try:
            response2 = llm.chat.completions.create(
                model=model,
                messages=messages + [{"role": "assistant", "content": response.choices[0].message.content}],
                max_tokens=20,
                temperature=0.0,
                top_p=1.0,
            )
            response.choices[0].message.content += "\n</think>\nThe final answer is: \boxed" + response2.choices[0].message.content
            response.usage.completion_tokens += response2.usage.completion_tokens
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    return dict(text=response.choices[0].message.content, token_num=response.usage.completion_tokens)