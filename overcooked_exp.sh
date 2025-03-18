Layouts=("asymmetric_advantages" "coordination_ring")
for layout in ${Layouts[@]}; do
    python Overcooked_benchmarking.py --layout $layout --max_new_tokens 2000 --token_per_tick 2000 --model_name wenyl/Overcooked-GRPO-Qwen-2.5-0.5B-Instruct
    python Overcooked_benchmarking.py --layout $layout --max_new_tokens 2000 --token_per_tick 2000 --model_name Qwen/Qwen2.5-0.5B-Instruct
done