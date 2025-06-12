import threading
from openai import OpenAI
from generate import generate
from anthropic import AnthropicVertex
class ApiThreadedLLMClient:
    def __init__(self, args):
        self.num_threads = 0
        self.lock = threading.Lock()
        self.api_keys = args.api_keys
        self.accum = [0] * len(args.api_keys)
        self.token_queue_len = [0] * len(args.api_keys)
        self.resp = [0] * len(args.api_keys)
 
        self.slow_llm = [0] * len(args.api_keys)
        self.fast_llm = [0] * len(args.api_keys)

        self.token_per_tick = args.token_per_tick
        self.method = args.method
        self.slow_model = args.slow_model
        self.slow_base_url = args.slow_base_url
        self.fast_model = args.fast_model
        self.fast_base_url = args.fast_base_url
 
    def add_new_thread(self, idx):
        self.lock.acquire()
        self.accum[idx] = 0
        self.token_queue_len[idx] = 0
        self.resp[idx] = ""
        if self.method != "fast":
            if "claude" in self.slow_model:
                self.slow_llm[idx] = AnthropicVertex(region="us-east5", project_id="gcp-multi-agent")
            else:
                self.slow_llm[idx] = OpenAI(base_url=self.slow_base_url, api_key=self.api_keys[idx])
                print(f"Thread {idx} using slow model: {self.slow_model} at {self.slow_base_url} with API key {self.api_keys[idx]}")
        if self.method != "slow":
            if "claude" in self.fast_model:
                self.fast_llm[idx] = AnthropicVertex(region="us-east5", project_id="gcp-multi-agent")
            else:                
                self.fast_llm[idx] = OpenAI(ase_url=self.slow_base_url, api_key=self.api_keys[idx])
            print(f"Thread {idx} using fast model: {self.fast_model} at {self.fast_base_url} with API key {self.api_keys[idx]}")
        self.lock.release()

    def generate(self, thread_id, messages, sampling_params, fast = False):
        if messages == []:
            return {"text": "", "token_num": 0}
        llm = self.slow_llm[thread_id] if not fast else self.fast_llm[thread_id]
        model = self.slow_model if not fast else self.fast_model
        return generate(llm, model, messages, sampling_params)
        
    def run_slow_inference(self, id, messages, DEFAULT_COMPLETION, sampling_params):
        """
        Run inference for the slow agent.
        If the messages is empty, means meta-control is false, and keep running previous inference.
        If the message is not empty, means meta-control is true, and drop the running inference.
        Return:
            - response text: DEFAULT_COMPLETION if the inference is not done yet
            - turn: when (how many turns before) the inference is initiated.
        """
           
        # none-empty messages: trigger slow agent, drop previous thinking
        if messages != []:
            self.accum[id] = 0
            self.token_queue_len[id] = 0
            self.resp[id] = ""
        turn = self.accum[id] // self.token_per_tick
        self.accum[id] += self.token_per_tick
        if self.token_queue_len[id] > 0:
            if self.token_queue_len[id] <= self.accum[id]:
                self.accum[id] = 0
                self.token_queue_len[id] = 0
                return self.resp[id], turn
            else:
                return DEFAULT_COMPLETION, turn
        response = self.generate(id, messages, sampling_params)
        self.resp[id] = response['text']
        self.token_queue_len[id] = response['token_num']
        if self.accum[id] >= self.token_queue_len[id]:
            self.accum[id] = 0
            self.token_queue_len[id] = 0
            return self.resp[id], turn
        else:
            return DEFAULT_COMPLETION, turn

    def run_fast_inference(self, id, messages, sampling_params):
        response = self.generate(id, messages, sampling_params, fast = True)
        return response["text"]