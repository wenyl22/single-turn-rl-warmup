LLM_SYSTEM_PROMPT = "You are a friendly chat assistant who is correct and brief at all times."

RULES = '''
1. 10 parallel freeways are numbered from 0 to 9. Freeway 0 is at the bottom and Freeway 9 is at the top. Each can be viewed as an axis in $x$ direction, on which cars travel in $+x$ or $-x$ direction.
2. I start at Freeway 9, and I want to end up in Freeway 0. In each turn, I can move one step, either up or down to the neighbouring freeway or stay.
3. I can only move vertically, which means my x position is fixed to be $x=0$.
4. Each car has different speed in $x$ direction. The speed is given as the number of turns it takes to move 1 unit forward. 
5. I bump into a car if we are in a same freeway and same $x$ position. If that happens, I will be sent back to the starting position on Freeway 9.
6. The episode ends if I get up to Freeway 0.
'''

BASE_PROMPT = f''' I am playing the game """Freeway""". Freeway has following rules: {RULES}. In short, my $x$ position is fixed to be 0. I am at Freeway 9 and want to get to Freeway 0 in minimal number of steps. In this process I need to avoid bumping into cars travelling on the freeways. Help me select my next action and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.'''


MESSAGE = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT},
    {"role": "user", "content": BASE_PROMPT},
]