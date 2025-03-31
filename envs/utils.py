import re
from fuzzywuzzy import process
import threading
import queue
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
    
    def generate(self, thread_id, messages):
        self.query_queues[thread_id].put(messages)
        return self.response_queues[thread_id].get()

    def run_inference(self, id, messages, STAY_COMPLETION):
        self.accum[id] += self.token_per_tick
        if self.token_queue_len[id] > 0:
            # dummy function call, indicating the thread is alive
            _ = self.generate(id, [])
            if self.token_queue_len[id] <= self.accum[id]:
                self.accum[id] = 0
                self.token_queue_len[id] = 0
                return self.resp[id]
            else:
                return STAY_COMPLETION
        response = self.generate(id, messages)
        self.resp[id] = response['text']
        self.token_queue_len[id] = len(response['token_ids'])
        if self.accum[id] >= self.token_queue_len[id]:
            self.accum[id] = 0
            self.token_queue_len[id] = 0
            return self.resp[id]
        else:
            return STAY_COMPLETION

        
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def find_best_match(action_string, available_actions_list, STAY_COMPLETION):
    if "</think>" not in action_string:
        action_string = STAY_COMPLETION
    else:
        action_string = action_string.split("</think>")[-1]
    if action_string == "":
        action_string = STAY_COMPLETION
    match = re.search(r'\\boxed\{(.+?)\}', action_string)
    if match:
        selected_match = match.group(1).strip()
    else:
        selected_match = action_string
    if len(selected_match) == 1 and selected_match.isalpha():
        return available_actions_list[ord(selected_match) - ord('A')]
    for action in available_actions_list:
        if selected_match.lower() in action.lower():
            return action 
    selected_move, score = process.extractOne(selected_match, available_actions_list)
    return selected_move
