import threading
import queue
from openai import OpenAI
from generate import generate_vanilla_openai, generate_prompted_s1_openai, generate_with_budget_reminder
from transformers import AutoTokenizer
class LocalThreadedLLMClient:
    def __init__(self, token_per_tick = 500):
        self.num_threads = 0
        self.lock = threading.Lock()
        self.query_queues = {}
        self.response_queues = {}
        self.accum = []
        self.token_queue_len = []
        self.resp = []
        self.token_per_tick = token_per_tick

    def add_new_thread(self):
        self.lock.acquire()
        self.num_threads += 1
        self.accum.append(0)
        self.token_queue_len.append(0)
        self.resp.append("")
        thread_id = self.num_threads - 1
        self.query_queues[thread_id] = queue.Queue()
        self.response_queues[thread_id] = queue.Queue()
        self.lock.release()
        return thread_id
    
    def generate(self, thread_id, messages, sampling_params):
        self.query_queues[thread_id].put((messages, sampling_params))
        return self.response_queues[thread_id].get()

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

class ApiThreadedLLMClient:
    def __init__(self, token_per_tick, args):
        self.num_threads = 0
        self.lock = threading.Lock()
        self.accum = []
        self.token_queue_len = []
        self.resp = []
        self.token_per_tick = token_per_tick
        self.llm = {}
        self.api_keys = args.api_keys
        self.base_url = args.base_url
        self.model = args.model
        if args.model == "deepseek-reasoner":
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        elif args.model == "deepseek-chat":
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(args.model)
            except:
                raise ValueError(f"Model {args.model} not found.")
        self.budget_forcing = args.budget_forcing
        self.query_queues = {}

    def add_new_thread(self):
        self.lock.acquire()
        self.num_threads += 1
        self.accum.append(0)
        self.token_queue_len.append(0)
        self.resp.append("")
        thread_id = self.num_threads - 1
        assert thread_id < len(self.api_keys), "Not enough API keys provided"
        self.llm[thread_id] = OpenAI(api_key=self.api_keys[thread_id], base_url=self.base_url)
        print(f"Thread {thread_id} initialized with API key {self.api_keys[thread_id]}, base URL {self.base_url}")
        self.lock.release()
        return thread_id

    def generate(self, thread_id, messages, sampling_params):
        if messages == []:
            return None
        if self.budget_forcing == "no":
            return generate_vanilla_openai(self.llm[thread_id], self.tokenizer, self.model, messages, sampling_params)
        elif self.budget_forcing == "ps":
            return generate_prompted_s1_openai(self.llm[thread_id], self.tokenizer, self.model, messages, sampling_params)
        elif self.budget_forcing == "br":
            return generate_with_budget_reminder(self.llm[thread_id], self.tokenizer, self.model, messages, sampling_params)
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



