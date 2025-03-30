from vllm import LLM, SamplingParams
from typing import List, Dict
from transformers import PreTrainedTokenizer
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
                m["content"] += f" Think in less than {budget} tokens."
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
            prompt2.append(output.outputs[0].text + "\n</think>\n The final answer is: \\boxed")
    sampling_params = SamplingParams(max_tokens=20, min_tokens = 0, temperature=0.0, stop_token_ids = [tokenizer.special_tokens_map["eos_token"]])
    outputs2 = llm.generate(prompt2, sampling_params)
    index = 0
    for output in outputs:
        if "</think>" not in output.outputs[0].text:
            output.outputs[0].text += "\n</think>\n The final answer is: \\boxed" + outputs2[index].outputs[0].text
            index += 1
    return outputs
        