from envs.utils import generate_gif_from_string_map
import pandas as pd
### Example usage
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
log_file = 'logs/asterix/DeepSeek-R1-Distill-Qwen-1.5B/2025-04-01-15-01-47_no_8192_1004.csv'
df = pd.read_csv(log_file)
string_map_list = df['render'].tolist()
gif_path = log_file.replace('.csv', '.gif')
generate_gif_from_string_map(string_map_list, gif_path, font_path=font_path, font_size=20)