from vllm import SamplingParams
from typing import List, Dict
from openai import OpenAI
from anthropic import AnthropicVertex
from google import genai
import time
# from together import Together

def generate(llm: OpenAI | AnthropicVertex | genai.Client, model: str, messages: List[Dict], sampling_params: SamplingParams, fast: bool) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "timeout": 600,
    }
    while True:
        try:
            if isinstance(llm, AnthropicVertex):
                response = llm.messages.create(**params)
                text = response.content[0].text
                token_num = response.usage.output_tokens
            else:
                response = llm.chat.completions.create(**params)
                if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content != None:
                    text = '<think>' + response.choices[0].message.reasoning_content + "\n</think>\n" + response.choices[0].message.content
                else:
                    text = response.choices[0].message.content
                token_num = response.usage.completion_tokens
            return dict(text=text, token_num=token_num)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)