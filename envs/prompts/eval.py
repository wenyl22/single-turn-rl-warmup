EVAL_PROMPT = """TASK: Extract the selected choice (e.g., A, B, C) from an LLM's answer to a multiple-choice question.

INPUT FORMAT:
- answer_string: A text description of LLM's answer.
- choice_list: A list of options labeled with letters (A, B, C, etc.)

ANALYSIS REQUIREMENTS:
1. If the answer_string explicitly mentions some letter (e.g., "A", "B", "C"), select that option.
2. Identify semantic similarities between the answer_string and each available option. Choose the option that best matches the answer_string based on the analysis.
3. If answer_string doesn't make a clear choice, return 'Z' (indicating no valid choice).

OUTPUT FORMAT:
Brief explanation of your reasoning process
SINGLE letter (A/B/C/...) on the final line ONLY with no additional text"""

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
