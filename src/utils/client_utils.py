import threading
from openai import OpenAI
from generate import generate, generate_s1
from anthropic import AnthropicVertex
from transformers import AutoTokenizer
from google import genai
import time
class LLMClient:
    def __init__(self, args, api_keys):
        self.api_keys = api_keys
        self.to_flush = ""
        self.to_flush_turn = 0
        self.gen_accum = 0
        self.gen_text = ""
        self.gen_token = []
        self.gen_token_num = 0
        self.gen_turn = 0
        self.turns = []
        self.slow_llm = None
        self.fast_llm = None

        self.token_per_tick = args.token_per_tick
        self.fast_max_token = args.fast_max_token
        self.method = args.method
        self.format = args.format
        self.slow_model = args.slow_model
        self.slow_base_url = args.slow_base_url
        self.fast_model = args.fast_model
        self.fast_base_url = args.fast_base_url
        self.budget_method = args.budget_method
        self.add_new_thread(0)

    def add_new_thread(self, idx):
        if self.method != "fast":
            if "claude" in self.slow_model:
                self.slow_llm = AnthropicVertex(region="us-east5", project_id="gcp-multi-agent")
            elif "gemini" in self.slow_model:
                self.slow_llm = genai.Client(vertexai=True, project="gcp-multi-agent", location="global")
            else:
                self.slow_llm = OpenAI(base_url=self.slow_base_url, api_key=self.api_keys)
                print(f"Using slow model: {self.slow_model} at {self.slow_base_url} with API key {self.api_keys}")
            if self.slow_model == "deepseek-reasoner":
                self.slow_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        if self.method != "slow":
            if "claude" in self.fast_model:
                self.fast_llm = AnthropicVertex(region="us-east5", project_id="gcp-multi-agent")
            elif "gemini" in self.fast_model:
                self.fast_llm = genai.Client(vertexai=True, project="gcp-multi-agent", location="global")
            else:
                self.fast_llm = OpenAI(base_url=self.fast_base_url, api_key=self.api_keys)
                print(f"Using fast model: {self.fast_model} at {self.fast_base_url} with API key {self.api_keys}")

    def generate(self, messages, sampling_params, fast = False):
        if messages == []:
            return {"text": "", "token_num": 0}
        llm = self.slow_llm if not fast else self.fast_llm
        model = self.slow_model if not fast else self.fast_model
        if self.budget_method == "s1":
            return generate_s1(llm, model, messages, sampling_params, fast)
        if self.budget_method == "constrainedCoT": # Let’s think a bit step by step and limit the answer length to xxx words.
            return generate_constrainedCoT(llm, model, messages, sampling_params, fast)
        # SoT: 3 formats
        # Concise CoT: Be concise
            
        return generate(llm, model, messages, sampling_params, fast)
    def run_slow_inference(self, messages, sampling_params, turn):
        _token_num = 0
        # new input, i.e. trigger slow system
        if messages != []:
            self.gen_turn = turn
            self.gen_accum = -self.fast_max_token
            response = self.generate(messages, sampling_params)
            self.gen_text = response['text']
            self.gen_token_num = response['token_num']
            if self.format == "T":
                self.gen_token = self.slow_tokenizer.encode(self.gen_text)
            _token_num = self.gen_token_num
        # generate self.token_per_tick tokens
        self.gen_accum += self.token_per_tick
        # prepare flush content
        can_flush = self.gen_accum >= self.gen_token_num or self.format == "T"
        if can_flush:
            self.to_flush_turn = self.gen_turn
            if self.gen_accum >= self.gen_token_num:
                self.to_flush = self.gen_text
            elif self.format == "T":
                self.to_flush = self.slow_tokenizer.decode(self.gen_token[:self.gen_accum], skip_special_tokens=True)
        _text = self.to_flush
        _turn = self.to_flush_turn
        self.to_flush = ""
        # check for completion: set the system in idle state
        if self.gen_accum + self.fast_max_token >= self.gen_token_num:
            self.gen_text = ""
        return _text, _turn, _token_num

    def run_fast_inference(self, messages, sampling_params, ALL_ACTIONS, DEFAULT_ACTION):
        response = self.generate(messages, sampling_params, fast = True)
        text = response['text']
        token_num = response['token_num']
        if "oxed" in text:
            return text, token_num
        text += "\n Therefore, the final answer is \\boxed{"
        new_message = messages.copy() + [{"role": "assistant", "content": text}]
        max_attempt = 3
        while max_attempt > 0:
            max_attempt -= 1
            try:
                response = self.fast_llm.chat.completions.create(
                    model=self.fast_model, messages=new_message, max_tokens=1, temperature=0, top_p=1,
                )
                if response.choices[0].message.content.strip()[0] in ALL_ACTIONS:
                    text += response.choices[0].message.content.strip()[0] + '}'
                    break
            except Exception as e:
                time.sleep(1)
            if max_attempt == 0:
                text += DEFAULT_ACTION + '}'
        return text, token_num

class WallTimeLLMClients:
    def __init__(self, args, api_keys):
        self.fast_model = args.fast_model
        self.slow_model = args.slow_model
        self.fast_base_url = args.fast_base_url
        self.slow_base_url = args.slow_base_url
        self.fast_llm = OpenAI(base_url=self.fast_base_url, api_key=api_keys)
        self.slow_llm = OpenAI(base_url=self.slow_base_url, api_key=api_keys)

    def generate(self, messages, sampling_params, fast = False):
        client = self.fast_llm if fast else self.slow_llm
        model = self.fast_model if fast else self.slow_model
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "stream": True,
        }
        #print("Messages:", messages)
        response = client.chat.completions.create(**params)
        return response