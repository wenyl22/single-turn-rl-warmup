EVAL_PROMPT = """TASK: You are given an input string, which is an answer to a multi-choice question. Pleaseanalyze an input string and select the most appropriate option from a list of available choices.

INPUT FORMAT:
- choices_list: A list of options labeled with letters (A, B, C, etc.)
- input_string: A text description of the answer to a question.

ANALYSIS REQUIREMENTS:
1. Compare the input string with each option in the choices_list. Identify semantic similarities between the input_string and each available option. Choose the option that best matches the input_string.
2. If the input string seems incomplete or does not match any option, output "Z".

OUTPUT FORMAT:
{a SINGLE character (A/B/C/...)}"""