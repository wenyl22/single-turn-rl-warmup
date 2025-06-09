### Run benchmarks
```bash
python src/run.py --game [name] --difficulty [E/M/H] --seed_num [1-8] \
    --token_per_tick [] \
    --method [slow/fast/parallel] \
    --api_keys [your_api_keys] \
    --fast_model deepseek-chat --fast_base_url https://api.deepseek.com/beta  \
    --slow_model deepseek-reasoner --slow_base_url https://api.deepseek.com/beta
```