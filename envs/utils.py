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

EVAL_PROMPT = """
You are a specialized AI that extracts the selected answer choice (e.g., A, B, C, D) from an LLM's response to a multiple-choice question. Ignore explanations, reasoning, or other text—only extract the final chosen option. Follow these instructions precisely:

## Task Description
Given:
1. An `answer_string` - a string that may contain or describe a choice
2. An `choice_list` - a list of possible choices, each labeled with a capital letter

Your job is to determine which choice from the choice_list best matches the answer_string, and return ONLY the capital letter (A/B/C/etc.) of that choice.

## Rules for Matching:
- If the answer_string contains a capital letter followed by a period (like "A." or "B."), extract that letter
- If the answer_string contains a boxed capital letter (like "\\boxed{A}" or "\\boxed{B}"), extract that letter
- If the answer_string doesn't contain an explicit letter but describes an action that matches one in the choice_list, return the letter of the matching action
- If no match can be found, return 'Z'
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "answer_string: \"\\boxed{B}\"\nchoice_list: [\"A. Move up to Freeway 4\", \"B. Move down to Freeway 2\",  \"C. Stay in the same freeway\"]"
    },
    {"role": "assistant", "content": "B"},
    {
        "role": "user",
        "content": "answer_string: \"Stay\"\nchoice_list: [\"A. Move up to Freeway 2\", \"B. Move down to Freeway 0\",  \"C. Stay in the same freeway\"]"
    },
    {"role": "assistant", "content": "C"},
    {
        "role": "user",
        "content": "answer_string: \"Let's make a left turn here\"\nchoice_list: [\"A. Turn right\", \"B. Go straight\", \"C. Turn around\"]"
    },
    {"role": "assistant", "content": "Z"}
]


def model_match(client, thread_id, examples):
    """
    This function extract the selected choice from a given answer to a multi-choice question.
    Args:
        examples (list): List of examples containing the answer and the choices.
            answer (str): The answer string containing the selected choice.
            choice_list (list): List of choices available.
        DEFAULT_COMPLETION (str): Default choice if no match is found.
    Returns:
        selected_letters (char): The selected chocie (capital letter). If no match is found, returns 'Z'.
    """
    for example in examples:
        messages = [{"role": "system", "content": EVAL_PROMPT}]
        messages += FEW_SHOT_EXAMPLES
        answer_string = example['answer_string']
        choice_list = str(example['choice_list'])
        messages.append({
            "role": "user",
            "content": f"answer_string: \"{answer_string}\"\nchoice_list: {choice_list}"
        })
    print("Answer string:", messages[-1]["content"])
    selected_letter = client.generate(thread_id, messages)['text']
    print("Extractor output:", selected_letter)
    if "</think>" in selected_letter:
        selected_letter = selected_letter.split("</think>")[-1]
    selected_letter = extract_boxed(selected_letter)
    while ord(selected_letter[0]) < ord('A') or ord(selected_letter[0]) > ord('Z'):
        selected_letter = selected_letter[1:]
    print("Selected letter:", selected_letter[0])
    return selected_letter[0]


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

    def run_inference(self, id, messages, STAY_COMPLETION, ignore_tick_limit=False):
        if ignore_tick_limit:
            # 不走累积 token 的逻辑，直接生成
            response = self.generate(id, messages)
            return response['text']
        
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
        self.token_queue_len[id] = response['token_num']
        if self.accum[id] >= self.token_queue_len[id]:
            self.accum[id] = 0
            self.token_queue_len[id] = 0
            return self.resp[id]
        else:
            return STAY_COMPLETION

# def find_best_match(action_string, available_actions_list, STAY_COMPLETION):
    # if "</think>"  in action_string:
    #     action_string = action_string.split("</think>")[-1]
    # if action_string == "":
    #     action_string = STAY_COMPLETION
    # match = re.search(r'\\boxed\{(.+?)\}', action_string)
    # if match:
    #     selected_match = match.group(1).strip() 
    # else:
    #     selected_match = action_string
    # # Model may output words like '\boxed{A. Move up.}'. So we need to remove choose the first letter
    # selected_match = selected_match[0]
    # if selected_match.isalpha():
    #     if ord(selected_match) - ord('A') < len(available_actions_list):
    #         return available_actions_list[ord(selected_match) - ord('A')]
    #     else:
    #         return STAY_COMPLETION
    # for action in available_actions_list:
    #     if selected_match.lower() in action.lower():
    #         return action 
    # selected_move, score = process.extractOne(selected_match, available_actions_list)
    # return selected_move   

def extract_boxed(text):
    match = re.search(r'\\boxed{([^}]*)}', text)
    if match:
        text = match.group(1)
    match = re.search(r'\\text{([^}]*)}', text)
    if match:
        text = match.group(1)
    return text

 
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
    line_height = font_size + 8
    img_width = max_width * font_size // 2 + 80
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
