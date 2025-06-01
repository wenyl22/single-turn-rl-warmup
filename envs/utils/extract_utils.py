import re
from envs.prompts.eval import EVAL_PROMPT
from vllm import SamplingParams
def extract_boxed(text, default_value=""):
    """
    Extracts the \boxed{...} text from the input string.
    """
    pattern = r'oxed{' 
    start_index = text.rfind(pattern)
    if start_index == -1:
        return default_value if default_value else text.strip()
    start_index += len(pattern) - 1
    stack = []
    for i in range(start_index, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            if stack:
                stack.pop()
            if not stack:
                return text[start_index + 1:i].strip()
    return default_value if default_value else text[start_index + 1:].strip()

def extract_scratch_pad(text, old_scratch_pad, valid_actions=None):
    scratch_pad = extract_boxed(text.split("</think>")[-1], default_value="").strip()
    if scratch_pad != "":
        scratch_pad = re.sub(r'[^' + valid_actions + ']', '', scratch_pad)
        if scratch_pad != "":
            old_scratch_pad = scratch_pad
    return old_scratch_pad
        
def model_match(client, thread_id, example):
    if len(example['answer_string']) == 1 and example['answer_string'].isalpha():
        # dummy function call, indicating the thread is alive
        _ = client.generate(thread_id, [], None)
        return example['answer_string']
    for choice in example['choice_list']:
        if example['answer_string'].lower() in choice.lower():
            # dummy function call, indicating the thread is alive
            _ = client.generate(thread_id, [], None)
            return choice[0]
    messages = [{"role": "system", "content": EVAL_PROMPT}]
    answer_string = example['answer_string']
    choice_list = str(example['choice_list'])
    sampling_params = SamplingParams(temperature = 0.1, max_tokens= 1024)
    messages.append({
        "role": "user", "content": f"CHOICE_LIST: \"{choice_list}\"\nINPUT_STRING: {answer_string}"
    })
    print(messages)
    selected_letter = client.generate(thread_id, messages, sampling_params)['text']
    print(f"selected_letter: {selected_letter}")
    selected_letter = extract_boxed(selected_letter).split("</think>")[-1].strip()
    selected_letter = selected_letter.split("Answer")[-1].strip()
    selected_letter = re.sub(r'[^a-zA-Z]', '', selected_letter)
    if selected_letter == "":
        selected_letter = "Z"
    return selected_letter[0]


def find_best_match(client, thread_id, answer_string, choice_list, DEFAULT_COMPLETION):
    answer_string = extract_boxed(answer_string)
    if "</think>" in answer_string:
        answer_string = answer_string.split("</think>")[-1].strip()
    letter = model_match(client, thread_id, {'answer_string': answer_string, 'choice_list': choice_list})[0]
    if letter.isalpha() and ord(letter) - ord('A') < len(choice_list):
        return choice_list[ord(letter) - ord('A')]
    else:
        return DEFAULT_COMPLETION