from vllm import SamplingParams
from typing import List, Dict
from openai import OpenAI
from anthropic import AnthropicVertex
from google import genai
import time
import tiktoken

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
            # if isinstance(llm, AnthropicVertex):
            #     response = llm.messages.create(**params)
            #     text = response.content[0].text
            #     token_num = response.usage.output_tokens
            # else:
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

def generate_s1(llm: OpenAI | AnthropicVertex | genai.Client, model: str, messages: List[Dict], sampling_params: SamplingParams, fast: bool) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "timeout": 600,
    }
    assert not fast, "This function is for slow model only."
    while True:
        try:
            response = llm.chat.completions.create(**params)
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content != None:
                text = '<think>' + response.choices[0].message.reasoning_content + "\n</think>\n" + response.choices[0].message.content
            else:
                text = response.choices[0].message.content
            if "</think>" not in text:
                text = text + "\n</think>"
            elif "oxed" in text.split("</think>")[-1]:
                return text, token_num
            text += '\n Therefore, the final answer is: \\boxed{'
            new_message = messages.copy() + [{"role": "assistant", "content": text}]
            response = llm.chat.completions.create(
                model=model, messages=new_message, max_tokens=200, temperature=0, top_p=1,
            )
            text += response.choices[0].message.content + '}'
            token_num = response.usage.completion_tokens
            return dict(text=text, token_num=token_num)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)