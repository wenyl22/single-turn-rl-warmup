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

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "answer_srting: \"C\"\nchoice_list: [\"A. Move up to Freeway 4\", \"B. Move down to Freeway 2\",  \"C. Stay in the same freeway\"]"
    },
    {
        "role": "assistant",
        "content": "C"
    },
    {
        "role": "user",
        "content": "answer_srting: \"The final answer is \\boxed{B}\"\nchoice_list: [\"A. Move up to Freeway 4\", \"B. Move down to Freeway 2\",  \"C. Stay in the same freeway\"]"
    },
    {
        "role": "assistant",
        "content": "B"
    },
    {
        "role": "user",
        "content": "answer_srting: \"down\"\nchoice_list: [\"A. Move up to Freeway 2\", \"B. Move down to Freeway 0\",  \"C. Stay in the same freeway\"]"
    },
    {
        "role": "assistant",
        "content": "B"
    },
    {
        "role": "user",
        "content": "answer_srting: \"Let's make a left turn here\"\nchoice_list: [\"A. Turn right\", \"B. Go straight\", \"C. Turn around\"]"
    },
    {
        "role": "assistant",
        "content": "Z"
    },
    {
        "role": "user",
        "content": "answer_srting: \"5\"\nchoice_list: [\"A. Move up to Freeway 5\", \"B. Move down to Freeway 3\",  \"C. Stay in the same freeway\"]"
    },
    {
        "role": "assistant",
        "content": "A"
    }
]
