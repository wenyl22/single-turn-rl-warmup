from vllm import LLM, SamplingParams
from typing import List, Dict
from transformers import PreTrainedTokenizer
from openai import OpenAI
from tqdm import tqdm
import time
import re
def generate_vanilla(llm: LLM, tokenizer: PreTrainedTokenizer, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    Generate with no budget forcing.
    """
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

def generate_vanilla_openai(llm: OpenAI, tokenizer: PreTrainedTokenizer, model: str, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    Generate with no budget forcing.
    """
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p
    }

    if 'QWEN3' in model.upper():  
        params["extra_body"] = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True}
        }
    while True:
        try:
            response = llm.chat.completions.create(**params)
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content != None:
                return dict(text='<think>' + response.choices[0].message.reasoning_content + "\n</think>\n" + response.choices[0].message.content, token_num=response.usage.completion_tokens)
            else:
                return dict(text=response.choices[0].message.content, token_num=response.usage.completion_tokens)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

def generate_prompted_s1_openai(llm: OpenAI, tokenizer: PreTrainedTokenizer, model: str, messages: List[Dict], sampling_params: SamplingParams) -> str:
    remote = '/' not in model
    for m in messages:
        if m["role"] == "user":
            m["content"] += f" Think in less than {sampling_params.max_tokens - 30} tokens."
    eos = tokenizer.special_tokens_map["eos_token"]
    generation = ""
    token_used = 0
    for i in range(2):
        new_messages = messages.copy() + [{"role": "assistant", "content": generation, "prefix": True}]
        params = {
            "model": model,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p
        }

        if 'QWEN3' in model.upper():  
            params["extra_body"] = {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            }

        while True:
            try:
                if remote:
                    params["messages"] = new_messages
                    response = llm.chat.completions.create(**params)
                else:
                    params["prompt"] = tokenizer.apply_chat_template(new_messages, add_generation_prompt=True, tokenize=False) + generation
                    response = llm.completions.create(**params)
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        if not remote:
            response_text = response.choices[0].text
        elif not hasattr(response.choices[0].message, 'reasoning_content') or \
        response.choices[0].message.reasoning_content is None:
            response_text = response.choices[0].message.content
        else:
            response_text = "<think>" + response.choices[0].message.reasoning_content + "</think>" + response.choices[0].message.content
        response_tokens = tokenizer(response_text)["input_ids"]
        if len(response_tokens) > sampling_params.max_tokens:
            response_tokens = response_tokens[:sampling_params.max_tokens]
        response_tokens = response_tokens[:sampling_params.max_tokens]
        if i == 0:
            token_used += len(response_tokens)
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        generation += response_text
        if eos in response_text or (re.findall(r"oxed{([^}]*)}", response_text.split("</think>")[-1]) and "</think>" in response_text):
            break
        if i == 0:
            generation += "\n</think>\n In summary, the final answer is: \\boxed"
            sampling_params = SamplingParams(max_tokens=200, temperature=0.0, stop_token_ids=[tokenizer.special_tokens_map["eos_token"]])
    return dict(text=generation, token_num=token_used)

def generate_with_budget_reminder(llm: OpenAI, tokenizer: PreTrainedTokenizer, model: str, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    Remind the model to be concise after a fixed number of tokens.
    """
    remote = '/' not in model
    token_per_generation = 120 # tunable parameter
    max_generation = sampling_params.max_tokens // token_per_generation
    token_used = 0
    generation = ""
    eos = tokenizer.special_tokens_map["eos_token"]
    for i in range(max_generation):
        new_messages = messages.copy() + [{"role": "assistant", "content": generation, "prefix": True}]
        params = {
            "model": model,
            "max_tokens": token_per_generation,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p
        }
        if 'QWEN3' in model.upper():
            params["extra_body"] = {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        while True:
            try:
                if remote:
                    params["messages"] = new_messages
                    response = llm.chat.completions.create(**params)
                else:
                    params["prompt"] = tokenizer.apply_chat_template(new_messages, add_generation_prompt=True, tokenize=False) + generation
                    response = llm.completions.create(**params)
                    
                #print(response)
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        if not remote:
            response_text = response.choices[0].text
        elif not hasattr(response.choices[0].message, 'reasoning_content') or \
        response.choices[0].message.reasoning_content is None:
            response_text = response.choices[0].message.content
        else:
            response_text = "<think>" + response.choices[0].message.reasoning_content + "</think>" + response.choices[0].message.content
        response_tokens = tokenizer(response_text)["input_ids"]
        if len(response_tokens) > token_per_generation:
            response_tokens = response_tokens[:token_per_generation]
        response_tokens = response_tokens[:token_per_generation]
        token_used += len(response_tokens)
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        generation += response_text
        if eos in response_text or ("</think>" in response_text and re.findall(r"oxed{([^}]*)}", response_text)):
            break
        if i == max_generation - 2:
            generation += "\n</think>\n In summary, the final answer is: \\boxed"
        else:
            generation += f"(There are only {token_per_generation * (max_generation - i - 2)} tokens left to use. I must be more concise.)\n"

    return dict(text=generation, token_num=token_used)

def generate_with_state_interruption(llm: OpenAI, tokenizer: PreTrainedTokenizer, model: str, messages: List[Dict], sampling_params: SamplingParams) -> str:
    """
    Response truncation.
    """
    # sys+user+assis+...
    if len(messages) >= 7:
        return True, generate_prompted_s1_openai(llm, tokenizer, model, messages, sampling_params)
    params = {
        "model": model,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p
    }
    if 'QWEN3' in model.upper():
        params["extra_body"] = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True}
        }

    while True:
        try:
            response = llm.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
            )
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content != None:
        response_text = "<think>" + response.choices[0].message.reasoning_content + "</think>" + response.choices[0].message.content
    else:
        response_text = response.choices[0].message.content
    response_tokens = tokenizer(response_text)["input_ids"]
    
    if len(response_tokens) <= sampling_params.max_tokens:
        return True, dict(text=response_text, token_num=len(response_tokens))
    response_tokens = response_tokens[:sampling_params.max_tokens]
    truncated_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    if "</think>" in truncated_response:
        return True, dict(text=response_text, token_num=len(response_tokens))
    if "<think>" in truncated_response:
        truncated_response += "</think>"
    return False, dict(text=truncated_response, token_num=len(response_tokens))