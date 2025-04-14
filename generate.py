from vllm import LLM, SamplingParams
from typing import List, Dict
from transformers import PreTrainedTokenizer
from openai import OpenAI
from tqdm import tqdm
import time
def generate_vanilla(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict], sampling_params: SamplingParams, budget: int = 0) -> str:
    """
    Generate with no budget forcing.
    """
    prompt = [tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False) for message in messages]
    outputs = llm.generate(prompt, sampling_params)
    return outputs

def generate_prompted(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict],sampling_params: SamplingParams, budget: int = 0) -> str:
    """
    Add "Think in less xxx tokens" after user prompt.
    """
    for message in messages:
        for m in message:
            if m["role"] == "user":
                m["content"] += f" Think in less than {budget - 30} tokens."
    prompt = [tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False) for message in messages]
    outputs = llm.generate(prompt, sampling_params)
    return outputs


def generate_s1(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict], sampling_params: SamplingParams, budget: int = 0) -> str:
    """
    s1 style budget forcing: after "budget" token is used, if thinking does not finish, add stop thinking token and output answer.
    """
    
    sampling_params.max_tokens = budget - 30
    
    prompt = [tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False) for message in messages]
    outputs = llm.generate(prompt, sampling_params)
    prompt2 = []
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            prompt2.append(output.outputs[0].text + "\n</think>\nThe final answer is: \\boxed")
    sampling_params = SamplingParams(max_tokens=20, min_tokens = 0, temperature=0.0, stop_token_ids = [tokenizer.special_tokens_map["eos_token"]])
    outputs2 = llm.generate(prompt2, sampling_params)
    index = 0
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            output.outputs[0].text += "\n</think>\nThe final answer is: \\boxed" + outputs2[index].outputs[0].text
            index += 1
    return outputs

def generate_prompted_s1(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict], sampling_params: SamplingParams, budget: int = 0) -> str:
    """
    Add "Think in less xxx tokens" after user prompt. 
    After "budget" token is used, if thinking does not finish, add stop thinking token and output answer.
    """
    
    sampling_params.max_tokens = budget - 30
    
    for message in messages:
        for m in message:
            if m["role"] == "user":
                m["content"] += f" Think in less than {budget - 30} tokens."
    
    prompt = [tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False) for message in messages]
    outputs = llm.generate(prompt, sampling_params)
    prompt2 = []
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            prompt2.append(output.outputs[0].text + "\n</think>\nThe final answer is: \\boxed")
    sampling_params = SamplingParams(max_tokens=20, min_tokens = 0, temperature=0.0, stop_token_ids = [tokenizer.special_tokens_map["eos_token"]])
    outputs2 = llm.generate(prompt2, sampling_params)
    index = 0
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            output.outputs[0].text += "\n</think>\nThe final answer is: \\boxed" + outputs2[index].outputs[0].text
            index += 1
    return outputs

def generate_api(client: OpenAI, messages: List[Dict], budget: int = 0) -> str:
    """
    Generate with openai client, with no budget forcing.
    """
    outputs = []
    for message in tqdm(messages):
        for r in range(20): # max retries
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=message,
                    stream=False,
                    max_tokens=budget - 30,
                    temperature=0.6,
                    top_p=0.95,
                )
                print(response)
                outputs.append((response.choices[0].message.reasoning_content + "\n</think>\n" + response.choices[0].message.content, [0]))
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
    return outputs
        