from envs.utils.visualize_utils import generate_gif_from_string_map
import pandas as pd
from envs.freeway import seed_mapping
### Example usage
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
log_file = 'logs/freeway/Qwen2.5-32B-Instruct/2025-04-23-19-16-11_ps_4096_{seed}.csv'
import os
if not os.path.exists(f'example_gifs/{log_file.split("/")[-2]}-ma'):
    os.mkdir(f'example_gifs/{log_file.split("/")[-2]}-ma')
for seed in range(8):
    df = pd.read_csv(log_file.format(seed=seed))
    real_seed = seed_mapping[seed][0]
    string_map_list = df['render'].tolist()
    gif_path = f'example_gifs/{log_file.split("/")[-2]}-ma' + f'/freeway_{real_seed}.gif'
    generate_gif_from_string_map(string_map_list, gif_path, font_path=font_path, font_size=20)