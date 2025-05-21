model: R1, V3, Distill-32B, Qwen-Instruct-32B

Critical-Step Planning:
prompt: math/game format, w/o predict state, w/o self-predict state
budget_forcing: ps, thought terminator
metric: plan safety, optimality, token count


```bash
python test_accuracy.py \
    --game freeway \
    --model model_name --api_keys "your_api_key" --base_url "url" \
    --max_new_tokens 8192/4096 --budget_forcing no/ps/tt \
    --prompt_format "math/game" --predict_state no/yes/self
1. --max_new_tokens 8192 --budget_forcing no --prompt_format "math/game" --predict_state no/yes/self -> 6 exps
2. --max_new_tokens 4096 --budget_forcing ps/tt --prompt_format "math/game" --predict_state no -> 4 exps
```
Conclusion:
1. LRMs need to understand the game rules in terms of math.
2. LRMs are bad at handling interruptions.


Full-Game Planning:
framework: high level agent, parallel high level & low level agent
prompt: math/game format
budget_forcing: ps, thought terminator
metric: game turn, total reward, plan safety

```bash
python run.py \
    --game freeway \
    --model model_name --api_keys "your_api_key" --base_url "url" \
    --framework sa/pma \
    --max_new_tokens 8192/4096 --token_per_tick 32768/4096 --budget_forcing no/ps/tt/si \
    --prompt_format "math"
1. --max_new_tokens 8192 --token_per_tick 32768 --budget_forcing no --prompt_format "math" --framework sa/pma -> 2 exps
2. --max_new_tokens 8192 --token_per_tick 4096 --budget_forcing no --prompt_format "math" --framework sa/pma -> 2 exps
3. --max_new_tokens 4096 --token_per_tick 4096 --budget_forcing ps --prompt_format "math" --framework sa/pma -> 4 exps
```

Conclusion:
1. LRMs alone are too slow for real-time games, need to cooperate with a low-level agent(rule-based / LLMs).
2. LRMs cannot properly handle interruptions.




High-level agent only:
    plan completely
Low-level agent only:
    plan for one step