import argparse
import datetime 
import os
import numpy as np
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed
from game_loop import main_game_loop

def check_args(args):
    if args.method == "slow":
        assert args.fast_max_token == 0, "Fast max token must be 0 when method is slow."
        assert args.format == "A", "Format must be 'A' when method is slow."
    if args.method == "fast":
        assert args.fast_max_token == args.token_per_tick, "Fast max token must be equal to token per tick when method is fast." 
    if args.method == "parallel":
        assert args.fast_max_token <= args.token_per_tick, "Fast max token must be less than or equal to token per tick when method is parallel." 
def jobs_to_schedule(Args):
    seed_num = 8
    instance_groupnum = 1
    instance_num = 8
    temp = []
    temp.extend(
        ['freeway-M-fast-32768-32768-A-1']
    )
    assert len(temp) == instance_groupnum, f"Expected {instance_groupnum} settings, got {len(temp)}"
    
    settings = temp.copy()
    instance = []
    for s in settings:
        repeat_times = int(s.split('-')[-1])
        game = s.split('-')[0]
        log_file = f"adaptive-logs-{game}/{s.replace('-', '_')[:-2]}"
        if not os.path.exists(log_file):
            os.makedirs(log_file)
        # make an argument instance
        args = argparse.Namespace(
            game=s.split('-')[0],
            difficulty=s.split('-')[1],
            method=s.split('-')[2],
            token_per_tick=int(s.split('-')[3]),
            fast_max_token=int(s.split('-')[4]),
            format=s.split('-')[5],
            repeat_times=repeat_times,
            slow_model=Args.slow_model,
            slow_base_url= Args.slow_base_url,
            fast_model= Args.fast_model,
            fast_base_url= Args.fast_base_url,
            meta_control= Args.meta_control,
            api_keys='to be assigned',
        )
        # check validity
        check_args(args)
        with open(log_file + '/args.log', 'w') as f:
            f.write("Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
            f.write("\n")
        for seed in range(seed_num):
            for r in range(repeat_times):
                if not os.path.exists(f"{log_file}/{r}_{seed}.csv"):
                    instance.append((log_file + f'/{r}_{seed}.csv', seed, args))
    # sort by r
    print(instance)
    assert len(instance) == instance_num, f"Expected {instance_num} instances, got {len(instance)}"
    instance.sort(key=lambda x: int(x[0].split('/')[-1].split('_')[0]))
    api_keys = Args.api_keys
    api_keys = api_keys * (instance_num // len(api_keys)) + api_keys[:instance_num % len(api_keys)]
    assert len(api_keys) == instance_num
    max_workers = len(api_keys)
    api_cycle = cycle(api_keys)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                main_game_loop, log_file, seed, args, next(api_cycle)
            )
            for (log_file, seed, args) in instance
        ]
        results = []
        total = len(futures)
       # write to the correct log file
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            with open(result['logdir']+'/args.log', 'a') as f:
                for key, value in result.items():
                    if key == 'logdir':
                        continue
                    f.write(f"{key}: {value} ")
                f.write("\n---------------------------------------\n")
            print(f"Progress: {idx}/{total} ({idx/total*100:.2f}%)")

if __name__ == "__main__":
    Args = argparse.ArgumentParser(description='Run benchmark with a specific model.')
    Args.add_argument('--api_keys', nargs='+', type=str, default=[], help='List of API keys for OpenAI')
    Args.add_argument('--slow_base_url', type=str, default=None, help='URL of the slow model server')
    Args.add_argument('--fast_base_url', type=str, default=None, help='URL of the fast model server')
    Args.add_argument('--slow_model', type=str, default = 'deepseek-reasoner')
    Args.add_argument('--fast_model', type=str, default = 'deepseek-chat')
    Args.add_argument('--meta_control', type=str, default='continuous', help='method to trigger slow agent')
    Args = Args.parse_args()
    jobs_to_schedule(Args)