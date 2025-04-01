import re
from fuzzywuzzy import process
import threading
import queue
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    img_width = max_width * font_size // 2 + 20
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
    draw.text((10, 0), index_text, fill="white", font=font)
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
