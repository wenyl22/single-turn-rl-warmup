from vllm import SamplingParams
from typing import List, Dict
from openai import OpenAI
import time

def generate(llm: OpenAI, model: str, messages: List[Dict], sampling_params: SamplingParams) -> str:
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