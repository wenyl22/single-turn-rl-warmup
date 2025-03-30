# Budget forcing
- Budget Forcing for efficient (real-time) reasoning Benchmark
    - No budget forcing ✅
    - Prompted CoT ✅
    - s1 style ✅
    - L1
        - Use pre-finetuned L1 model
        - reproduce/finetune L1 on our task
    - DeepScaleR?
- Two budget scheduling strategies for multi-step planning?
    - consistent
    - linear/exponential decay (intuition: planning more at the early steps)
# Experiments
- Different games
- Different distill models(include trained L1 models)
- Different budget forcing strategies
- Different budget tokens(max_tokens fixed to be, say 8k)
