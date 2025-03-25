import re
from fuzzywuzzy import process

def find_best_match(action_string, available_actions_list):
    if "</think>" not in action_string:
        action_string = "stay in the same freeway"
    else:
        action_string = action_string.split("</think>")[-1]
    if action_string == "":
        action_string = "stay in the same freeway"
    # search for \boxed{} and extract the content
    match = re.search(r'\\boxed\{(.+?)\}', action_string)
    if match:
        selected_match = match.group(1).strip()
    else:
        selected_match = action_string
    for action in available_actions_list:
        if selected_match.lower() in action.lower():
            return action 
    selected_move, score = process.extractOne(selected_match, available_actions_list)
    return selected_move

def accuracy_reward(completions, solution, available_actions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion for completion in completions]
    rewards = []
    for content, sol, av_action in zip(contents, solution, available_actions):
        select_action = find_best_match(content, av_action)
        if (select_action in av_action) and (select_action == sol):
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, and out of the thinking process, the final answer has the answer in \\boxed{}."""
    pattern = r".*?</think>.*?\\boxed\{.*?\}.*?"    
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    return [1.0 if match else 0.0 for match in matches]