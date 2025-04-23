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