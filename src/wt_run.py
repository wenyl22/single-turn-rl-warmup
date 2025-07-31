import argparse
import datetime 
import os
import numpy as np
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed
from wt_game_loop import GamePlay

def check_args(args):
    if args.method == "slow":
        assert args.fast_agent_time == 10, "Fast max token must be 0 when method is slow."
        assert args.format == "A", "Format must be 'A' when method is slow."
    if args.method == "fast":
        assert args.fast_agent_time == args.seconds_per_step, "Fast max token must be equal to token per tick when method is fast." 
    if args.method == "parallel":
        assert args.fast_agent_time <= args.seconds_per_step, "Fast max token must be less than or equal to token per tick when method is parallel." 
def jobs_to_schedule(Args):
    seed_num = 8
    instance_groupnum = 1
    instance_num = 8
    temp = []
    temp.extend(
        ['overcooked-M-parallel-360-90-T']
    )
    assert len(temp) == instance_groupnum, f"Expected {instance_groupnum} settings, got {len(temp)}"
    
    settings = temp.copy()
    instance = []
    for s in settings:
        game = s.split('-')[0]
        log_dir = f"walltime-logs/{s.replace('-', '_')}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # make an argument instance
        args = argparse.Namespace(
            game = s.split('-')[0],
            difficulty = s.split('-')[1],
            method = s.split('-')[2],
            seconds_per_step = int(s.split('-')[3]),
            fast_agent_time = int(s.split('-')[4]),
            format = s.split('-')[5],
            slow_model = Args.slow_model,
            slow_base_url = Args.slow_base_url,
            fast_model = Args.fast_model,
            fast_base_url = Args.fast_base_url,
            meta_control = Args.meta_control,
            api_keys = "to be assigned",
        )
        check_args(args)
        if not os.path.exists(log_dir + '/args.log'):
            with open(log_dir + '/args.log', 'w') as f:
                f.write("Arguments:\n")
                for arg, value in vars(args).items():
                    f.write(f"{arg}: {value}\n")
                f.write("\n")
        for seed in range(seed_num):
            if not os.path.exists(f"{log_dir}/game_{seed}.json"):
                instance.append((log_dir, seed, args))
    # sort by r
    print(instance)
    assert len(instance) == instance_num, f"Expected {instance_num} instances, got {len(instance)}"
    api_keys = Args.api_keys
    api_keys = api_keys * (instance_num // len(api_keys)) + api_keys[:instance_num % len(api_keys)]
    assert len(api_keys) == instance_num
    max_workers = len(api_keys)
    api_cycle = cycle(api_keys)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for logdir, seed, args in instance:
            args.api_keys = next(api_cycle)
            print(args.api_keys)
            futures.append(executor.submit(GamePlay().run_env, logdir, seed, args))
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
    for s in settings:
        log_dir = f"walltime-logs-{s.split('-')[0]}/{s.replace('-', '_')}"
        log_dir_results = [r for r in results if r['logdir'] == log_dir]
        with open(log_dir + '/args.log', 'a') as f:
            # write the summary of results, mean of time, turns, and reward
            f.write("\nSummary of results:\n")
            f.write(f"Total runs: {len(log_dir_results)}\n")
            for key, value in log_dir_results[0].items():
                if key == 'logdir':
                    continue
                mean_value = np.mean([r[key] for r in log_dir_results])
                f.write(f"{key}: {mean_value:.2f} ")
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