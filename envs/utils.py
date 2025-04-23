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

EVAL_PROMPT = """TASK: Analyze an input string and select the most appropriate option from a list of available choices.

INPUT FORMAT:
- choices_list: A list of options labeled with letters (A, B, C, etc.)
- input_string: A text description of an action

ANALYSIS REQUIREMENTS:
1. Analyze the input_string for key terms, direction indicators, and intent.
2. Compare the input_string with each option in the choices_list.
3. Identify semantic similarities between the input_string and each available option.

OUTPUT FORMAT:
Reasoning:
{Brief explanation of your reasoning process}
Answer:
{a SINGLE character (A/B/C/...)}"""

def find_best_match(client, thread_id, response, choices, default_completion):
    """
    This function finds the best match for a given response from a list of choices.
    Args:
        client: The LLM client to use for generating responses.
        thread_id: The ID of the thread to use for generating responses.
        response (str): The response string to match against the choices.
        choices (list): List of available choices.
        default_completion (str): Default choice if no match is found.
    Returns:
        selected_choice (str): The selected choice from the list of choices.
    """

    extract_response = response
    if "</think>" in extract_response:
        extract_response = extract_response.split("</think>")[-1]
    extract_response = extract_boxed(extract_response)

    input_string = "CHOICE_LIST: " + str(choices) + "\n" + "INPUT_STRING: " + extract_response
    print("Input string:", input_string)
    messages = [
        {"role": "system", "content": EVAL_PROMPT},
        {"role": "user", "content": input_string}
    ]
    response = client.generate(thread_id, messages)
    print("Extractor output:", response['text'])

    extract_letter = response['text'].split("\n")[-1].strip()
    no = ord(extract_letter[0]) - ord('A')
    if no < 0 or no >= len(choices):
        print("No valid letter found, defaulting to 'Z'")
        no = len(choices) - 1
    
    return choices[no]
    

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
        self.token_queue_len[id] = response['token_num']
        if self.accum[id] >= self.token_queue_len[id]:
            self.accum[id] = 0
            self.token_queue_len[id] = 0
            return self.resp[id]
        else:
            return STAY_COMPLETION

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
