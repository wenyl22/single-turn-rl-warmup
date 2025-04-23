import re
import threading
import queue
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from envs.prompts.eval import EVAL_PROMPT

    

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
    
    def generate(self, thread_id, messages, qry_type="reasoning"):
        self.query_queues[thread_id].put((messages, qry_type))
        return self.response_queues[thread_id].get()

    def run_inference(self, id, messages, DEFAULT_COMPLETION):        
        self.accum[id] += self.token_per_tick
        if self.token_queue_len[id] > 0:
            # dummy function call, indicating the thread is alive
            _ = self.generate(id, [])
            if self.token_queue_len[id] <= self.accum[id]:
                self.accum[id] = 0
                self.token_queue_len[id] = 0
                return self.resp[id]
            else:
                return DEFAULT_COMPLETION
        response = self.generate(id, messages)
        self.resp[id] = response['text']
        self.token_queue_len[id] = response['token_num']
        if self.accum[id] >= self.token_queue_len[id]:
            self.accum[id] = 0
            self.token_queue_len[id] = 0
            return self.resp[id]
        else:
            return DEFAULT_COMPLETION

def extract_boxed(text):
    """
    Extracts the \boxed{...} text from the input string.
    """
    pattern = r'\boxed{(.*?)}'
    matches = re.findall(pattern, text)
    # delete leading and trailing spaces
    if matches:
        return matches[0].strip()
    else:
        return text.strip()
        
def model_match(client, thread_id, example):
    if len(example['answer_string']) == 1 and example['answer_string'][0].isalpha():
        # dummy function call, indicating the thread is alive
        _ = client.generate(thread_id, [])
        return example['answer_string'][0]
    for choice in example['choice_list']:
        if example['answer_string'].lower() in choice.lower():
            # dummy function call, indicating the thread is alive
            _ = client.generate(thread_id, [])
            return choice
    messages = [{"role": "system", "content": EVAL_PROMPT}]
    answer_string = example['answer_string']
    choice_list = str(example['choice_list'])
    messages.append({
        "role": "user", "content": f"CHOICE_LIST: \"{choice_list}\"\nINPUT_STRING: {answer_string}"
    })
    selected_letter = client.generate(thread_id, messages)['text'].split("</think>")[-1].strip()
    if selected_letter == "":
        selected_letter = "Z"
    return selected_letter[0]


def find_best_match(client, thread_id, answer_string, choice_list, DEFAULT_COMPLETION):
    if "</think>" in answer_string:
        answer_string = answer_string.split("</think>")[-1]
    answer_string = extract_boxed(answer_string)
    letter = model_match(client, thread_id, {'answer_string': answer_string, 'choice_list': choice_list})
    if letter.isalpha() and ord(letter) - ord('A') < len(choice_list):
        return choice_list[ord(letter) - ord('A')]
    else:
        return DEFAULT_COMPLETION


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