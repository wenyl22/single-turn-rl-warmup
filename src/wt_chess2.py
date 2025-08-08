from envs.minatar.environments.chess2 import Env
from envs.minatar.environment import Environment
import time
from utils.client_utils import SimpleLLMClients
from utils.extract_utils import extract_boxed
from collections import defaultdict
import threading
import queue
from openai.types.responses import (
    ResponseReasoningSummaryTextDoneEvent,
    ResponseCompletedEvent,
    Response,
)
import pandas as pd
from envs.chess2 import llm_state_builder, state_to_description
from envs.prompts.chess2 import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT

class GamePlay:
    def __init__(self):
        pass
    def fast_agent_move(self, client, state_description, player_idx):
        assert isinstance(self.env.env, Env), "Environment must be an instance of Env"
        legal_moves = self.env.env.board.legal_moves
        prompt = FAST_AGENT_PROMPT + ACTION_FORMAT_PROMPT + state_description
        action = None
        _fast, _slow, text = 0, 0, ""
        while not action or action not in legal_moves:
            start_time = time.time()
            response = client.get_fast_response(prompt)
            assert isinstance(response, Response)
            _fast = response.usage.output_tokens
            text = "<think>" + response.output[0].content[0].text + "</think>" + response.output[1].content[0].text
            elapsed = time.time() - start_time
            self.time_used[player_idx] += elapsed
            timeout = self.time_used[player_idx] >= self.time_limits[player_idx]
            action = self.env.env._parse_action(extract_boxed(text.split("</think>")[-1].strip()))
            if timeout:
                break
        return text, timeout, _fast, _slow, action
    
    def slow_agent_move(self, client, state_description, player_idx):
        assert isinstance(self.env.env, Env), "Environment must be an instance of Env"
        legal_moves = self.env.env.board.legal_moves
        prompt = SLOW_AGENT_PROMPT + ACTION_FORMAT_PROMPT + state_description
        action = None
        _fast, _slow, text = 0, 0, ""
        while not action or action not in legal_moves:
            start_time = time.time()
            response = client.get_slow_response(prompt)
            assert isinstance(response, Response)
            _slow = response.usage.output_tokens
            text = "<think>" + response.output[0].content[0].text + "</think>" + response.output[1].content[0].text
            elapsed = time.time() - start_time
            self.time_used[player_idx] += elapsed
            timeout = self.time_used[player_idx] >= self.time_limits[player_idx]
            action = self.env.env._parse_action(extract_boxed(text.split("</think>")[-1].strip()))
            if timeout:
                break
        return text, timeout, _fast, _slow, action
    
    def parallel_agent_move(self, client, state_description, player_idx):
        assert isinstance(self.env.env, Env), "Environment must be an instance of Env"
        legal_moves = self.env.env.board.legal_moves
        action = None
        _fast, _slow, text = 0, 0, ""
        timeout = False
        def slow_llm_worker():
            try:
                prompt = SLOW_AGENT_PROMPT + CONCLUSION_FORMAT_PROMPT + state_description
                for chunk in client.get_slow_response_streaming(prompt, cancel_event):
                    if isinstance(chunk, ResponseReasoningSummaryTextDoneEvent):
                        chunk_queue.put(('chunk', chunk.text))
                    elif isinstance(chunk, ResponseCompletedEvent):
                        chunk_queue.put(('chunk', chunk.response.output[1].content[0].text))
                        chunk_queue.put(('usage', chunk.response.usage.output_tokens))
            except Exception as e:
                chunk_queue.put(('error', str(e)))

        while not action or action not in legal_moves:
            start_time = time.time()
            chunk_queue = queue.Queue()
            cancel_event = threading.Event()
            slow_thread = threading.Thread(target=slow_llm_worker)
            slow_thread.start()   
            accumulated_response = ""        
            while True:
                if time.time() - start_time + self.time_used[player_idx] >= self.time_limits[player_idx]:
                    cancel_event.set()
                    action = DEFAULT_ACTION
                    break
                time.sleep(0.5)
                while not chunk_queue.empty():
                    msg_type, content = chunk_queue.get()
                    if msg_type == 'chunk':
                        accumulated_response += content
                    elif msg_type == 'usage':
                        _slow = content
                state_description_with_advice = state_description + "\nGuidance from a Previous Thinking Model:\n>" + ">".join(accumulated_response.split('\n')) + "\n"
                fast_response = client.get_fast_response(state_description_with_advice)
                assert isinstance(fast_response, Response)
                _fast += fast_response.usage.output_tokens
                text += "<think>" + fast_response.output[0].content[0].text + "</think>" + fast_response.output[1].content[0].text
                action = self.env.env._parse_action(extract_boxed(text.split("</think>")[-1].strip()))
                if action and action in legal_moves:
                    break
        slow_thread.join(timeout=1.0)
        elapsed = time.time() - start_time
        self.time_used[player_idx] += elapsed
        timeout = self.time_used[player_idx] >= self.time_limits[player_idx]
        return "SLOW\n\n" + accumulated_response + "\n\nFAST\n\n" + text, timeout, _fast, _slow, action
 
    def run_env(self, log_dir, seed, args):
        self.time_limits = [args.time_limits, args.time_limits]
        self.time_used = [0.0, 0.0]
        env = Environment('chess2', sticky_action_prob=0.0)
        assert isinstance(env.env, Env), "Environment must be an instance of Env"
        self.env = env
        env.seed(seed)
        env.reset()
        methods = [args.player1_method, args.player2_method]
        client = SimpleLLMClients(args)
        logs = defaultdict(list)        
        while not env.env.terminal:
            logs['render'].append(env.env.state_string())
            player_idx = env.env.player_idx
            state_for_llm = llm_state_builder(env.env, self.time_limits[player_idx]- self.time_used[player_idx],
                                              self.time_limits[1 - player_idx] - self.time_used[1 - player_idx])
            state_description = state_to_description(state_for_llm)
            method = methods[player_idx]
            if method == 'fast':
                text, timeout, _fast, _slow, action = self.fast_agent_move(client, state_description, player_idx)
            elif method == 'slow':
                text, timeout, _fast, _slow, action = self.slow_agent_move(client, state_description, player_idx)
            elif method == 'parallel':
                text, timeout, _fast, _slow, action = self.parallel_agent_move(client, state_description, player_idx)
            logs['action'].append(action)
            logs['fast_token_usage'].append(_fast)
            logs['slow_token_usage'].append(_slow)
            logs['response'].append(text)
            logs['time_used'].append(self.time_used[player_idx])
            logs['timeout'].append(timeout)
            if timeout:
                print(f"Game ended: Player {player_idx} timed out")
                break
            reward, done = env.act(action)
            print(f"Player {player_idx} ({method}) used action: {action}")
            print(f"Time remaining: {self.time_limits[player_idx] - self.time_used[player_idx]:.1f}s")
            df = pd.DataFrame(logs)
            df.to_csv(f"{log_dir}/{seed}.csv")
        return {
            'logdir': log_dir,
            'seed': seed,
            'reward': env.env.reward,
        }