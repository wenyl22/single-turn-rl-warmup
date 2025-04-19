from envs.utils import generate_gif_from_string_map
import pandas as pd
### Example usage
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
log_file = 'logs/asterix_no/2025-04-03-07-37-23_no_8192_1001.csv'
df = pd.read_csv(log_file)
string_map_list = df['render'].tolist()
gif_path = log_file.replace('.csv', '.gif')
generate_gif_from_string_map(string_map_list, gif_path, font_path=font_path, font_size=20)