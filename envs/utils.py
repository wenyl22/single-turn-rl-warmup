import re
from fuzzywuzzy import process
import threading
import queue
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sampling_params = SamplingParams(temperature=0.0, 
                                max_tokens=1024, 
                                n=1,
                                top_p=1,
                                )

EVAL_PROMPT = """You are a specialized AI that extracts a capital letter choice (A/B/C/etc.) from an action string by matching it against a list of available actions. Follow these instructions precisely:

## Task Description
Given:
1. An `action_string` - a text that may contain or describe an action choice
2. An `available_actions_list` - a list of possible actions, each labeled with a capital letter

Your job is to determine which action from the available list best matches the action_string, and return ONLY the capital letter (A/B/C/etc.) of that action.

## Rules for Matching:
- If the action_string contains a capital letter followed by a period (like "A." or "B."), extract that letter
- If the action_string contains a boxed capital letter (like "\boxed{A}" or "\boxed{B}"), extract that letter
- If the action_string doesn't contain an explicit letter but describes an action that matches one in the available_actions_list, return the letter of the matching action
- If no match can be found, return 'Z'

## Output Format:
First line includes detailed analysis of the text.
Second line includes ONLY the matching capital letter without any additional text, explanation, or formatting.

## Your Task:
Analyze the `action_string` and the `available_actions_list`.
Determine which action matches and return only the capital letter."""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "action_string: \"\\boxed{B}\"\navailable_actions_list: [\"A. Move up to Freeway 4\", \"B. Move down to Freeway 2\",  \"C. Stay in the same freeway\"]"
    },
    {
        "role": "assistant",
        "content": "B"
    },
    {
        "role": "user",
        "content": "action_string: \"Stay in the same freeway\"\navailable_actions_list: [\"A. Move up to Freeway 2\", \"B. Move down to Freeway 0\",  \"C. Stay in the same freeway\"]"
    },
    {
        "role": "assistant",
        "content": "C"
    },
    {
        "role": "user",
        "content": "action_string: \"Let's make a left turn here\"\navailable_actions_list: [\"A. Turn right\", \"B. Go straight\", \"C. Turn around\"]"
    },
    {
        "role": "assistant",
        "content": "Z"
    }
]


def model_match(llm, tokenizer, examples):
    """
    This function extract the action from the action_string and match it with the available actions.
    Args:
        examples (list): List of examples containing the action string and available actions.
            action_string (str): The action string returned by the model.
            available_actions_list (list): List of available actions.
        STAY_COMPLETION (str): Default action if no match is found.
    Returns:
        selected_letters (char): The selected action (capital letter). If no match is found, returns 'Z'.
    """

    prompt_batch = []

    for example in examples:
        messages = [
            {
                "role": "system",
                "content": EVAL_PROMPT
            }
        ]
        messages += FEW_SHOT_EXAMPLES
        
        action_string = example['action_string']
        available_actions_list = str(example['available_actions_list'])
        
        messages.append({
            "role": "user",
            "content": f"action_string: \"{action_string}\"\navailable_actions_list: {available_actions_list}"
        })
        
        print("Checking completion:")
        print(messages[-1])
            
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_batch.append(prompt)


    completions = llm.generate(prompt_batch, sampling_params)
    

    selected_letters = []
    for i, completion in enumerate(completions):
        selected_letter = completion.outputs[0].text.strip()[-1]
        print("Response: ", completion.outputs[0].text)
        if selected_letter.isalpha():
            selected_letters.append(selected_letter)
        else:
            selected_letters.append('Z')
    return selected_letters


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

def find_best_match(llm, tokenizer, action_string, available_actions_list, STAY_COMPLETION):
    letter = model_match(llm, tokenizer, [{
        'action_string': action_string,
        'available_actions_list': available_actions_list
    }])[0]
    if letter.isalpha() and ord(letter) - ord('A') < len(available_actions_list):
        return available_actions_list[ord(letter) - ord('A')]
    else:
        return STAY_COMPLETION

def string_map_to_image(string_map, font_path, font_size, index):
    """
    Convert a string map to an image.

    Args:
        string_map (str): Text representation of the world.
        font_path (str): Path to a .ttf font file (optional).
        font_size (int): Font size for rendering the text.

    Returns:
        PIL.Image: Image object representing the string map.
    """
    lines = string_map.split('\n')
    max_width = max(len(line) for line in lines)
    line_height = font_size + 5
    img_width = max_width * font_size // 2 + 30
    img_height = len(lines) * line_height
    image = Image.new("RGB", (img_width, img_height), "black")
    draw = ImageDraw.Draw(image)
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)
    for i, line in enumerate(lines):
        draw.text((10, i * line_height), line, fill="white", font=font)
    index_text = f"Frame {index + 1}"
    draw.text((10, img_height - line_height), index_text, fill="white", font=font)
    return image

def generate_gif_from_string_map(string_map_list, gif_path, duration=1000, font_path=None, font_size=30):
    """
    Generate a .gif file from a list of string maps.

    Args:
        string_map_list (list): List of string maps.
        gif_path (str): Path to save the .gif file.
        duration (float): Duration for each frame in the gif.
        font_path (str): Path to a .ttf font file (optional).
        font_size (int): Font size for rendering the text.
    """
    images = []
 
    for i, string_map in enumerate(string_map_list):
        image = string_map_to_image(string_map, font_path, font_size, i)
        images.append(np.array(image))

    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved at {gif_path}")
    return gif_path
