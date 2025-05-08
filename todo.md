Experiment:
- token_per_tick = max_new_token, w & wo scratch pad, budget_forcing = s1
    - [] 14B & 32B, reasoning & Instruct model

Code Base:
- Budget Forcing:
    - [✔] s1: Interrupt within LLM thinking process. 
    - [] Thought Terminator: Interrupt with user prompt.
- Continuous planning: Agent hasn't completed planning in the last turn(only when `token_per_tick` < `max_new_tokens`). 
    - What action to take this turn:
        - [] Option1: Force a none-action.
        - [×] Option2: Extract generated plan to scratch pad and follow. 
            - Take away: Can not change thinking trajectory format
        - [] Option3: React.
    - What to do with the plan(which may arrive a few turns later):
        - [] Option1: Use it as the same[TODO].
        - [] Option2: Skip the turns that have passed.
        - [] Option3: Always up-to-date plan(interrupt with new game state).
- Use non-thinking model to react/follow the plan.
    - [] With scratch pad
    - [?] Without scratch pad