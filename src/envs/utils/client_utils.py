import threading
import queue
from openai import OpenAI
from generate import generate_vanilla_openai, generate_prompted_s1_openai, generate_with_budget_reminder, generate_with_state_interruption
from transformers import AutoTokenizer

class ApiThreadedLLMClient:
    def __init__(self, token_per_tick, args):
        self.num_threads = 0
        self.lock = threading.Lock()
        self.api_keys = args.api_keys
        self.accum = [0] * len(args.api_keys)
        self.token_queue_len = [0] * len(args.api_keys)
        self.resp = [0] * len(args.api_keys)
        self.message = [0] * len(args.api_keys)
        self.llm = [0] * len(args.api_keys)
        self.low_llm = [0] * len(args.api_keys)

        self.token_per_tick = token_per_tick
        self.model = args.model
        self.base_url = args.base_url

        self.method = args.method
        self.low_model = args.low_model
        self.low_base_url = args.low_base_url
        if args.low_model == "":
            pass
        elif args.low_model == "deepseek-reasoner":
            self.low_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        elif args.low_model == "deepseek-chat":
            self.low_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
        if args.model == "":
            pass
        elif args.model == "deepseek-reasoner":
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        elif args.model == "deepseek-chat":
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
        self.budget_forcing = args.budget_forcing

    def add_new_thread(self, idx):
        self.lock.acquire()
        self.accum[idx] = 0
        self.token_queue_len[idx] = 0
        self.resp[idx] = ""
        self.message[idx] = []
        if self.method != "lsa":
            self.llm[idx] = OpenAI(api_key=self.api_keys[idx], base_url=self.base_url)
        if self.method != "hsa":
            self.low_llm[idx] = OpenAI(api_key=self.api_keys[idx], base_url=self.low_base_url)
        print(f"Thread {idx} initialized with API key {self.api_keys[idx]} and base URL {self.base_url}")
        self.lock.release()

    def generate(self, thread_id, messages, sampling_params, low = False):
        if messages == []:
            return {"text": "", "token_num": 0}
        llm = self.llm[thread_id] if not low else self.low_llm[thread_id]
        model = self.model if not low else self.low_model
        tokenizer = self.tokenizer if not low else self.low_tokenizer
        if self.budget_forcing == "no":
            return generate_vanilla_openai(llm, tokenizer, model, messages, sampling_params)
        elif self.budget_forcing == "ps":
            return generate_prompted_s1_openai(llm, tokenizer, model, messages, sampling_params)
        elif self.budget_forcing == "br":
            return generate_with_budget_reminder(llm, tokenizer, model, messages, sampling_params)
        elif self.budget_forcing == "si":
            return generate_with_state_interruption(llm, tokenizer, model, messages, sampling_params)
        else:
            raise ValueError(f"Unsupported budget forcing method: {self.budget_forcing}")

    def run_inference(self, id, messages, DEFAULT_COMPLETION, sampling_params):   
        self.accum[id] += self.token_per_tick
        if self.token_queue_len[id] > 0:
            # dummy function call, indicating the thread is alive
            _ = self.generate(id, [], sampling_params)
            if self.token_queue_len[id] <= self.accum[id]:
                self.accum[id] = 0
                self.token_queue_len[id] = 0
                return self.resp[id]
            else:
                return DEFAULT_COMPLETION
        response = self.generate(id, messages, sampling_params)
        self.resp[id] = response['text']
        self.token_queue_len[id] = response['token_num']
        if self.accum[id] >= self.token_queue_len[id]:
            self.accum[id] = 0
            self.token_queue_len[id] = 0
            return self.resp[id]
        else:
            return DEFAULT_COMPLETION
    def run_inference_with_interruption(self, id, messages, DEFAULT_COMPLETION, sampling_params):
        # If the model has finished generating previously, open up a new message list.
        if self.message[id] == []:
            for m in messages:
                self.message[id].append({"role": m["role"], "content": m["content"]})
        else:
            content = "Wait. You think so long that one turn has passed before you act. \n" + messages[-1]["content"]
            self.message[id].append({"role": "user", "content": content})
        end, response = self.generate(id, self.message[id], sampling_params)
        if end:
            self.message[id] = []
        else:
            self.message[id].append({"role": "assistant", "content": response['text']})
        # print(f"-----------------Thread {id} -----------------")
        # print(f"Messages: {messages}")
        # print(f"-----------------------------------------------------------------------------")
        return end, response['text']
    def run_low_level_inference(self, id, messages, sampling_params):
        response = self.generate(id, messages, sampling_params, low = True)
        return response["text"]