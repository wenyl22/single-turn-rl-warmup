Label_Color_Mapping = {
    'Optimal': "#000000",
    'Think Fast': "#fa0d0d",
    'Parallel': "#ab4bce",
    'Think Slow': "#057bfa",
}

Number_Style_Mapping = {
    'default': 'o-',
    '4K': 'o-',
    '8K': '^-',
    '16K': 'x-',
    '32K': 's-',
}
import matplotlib.pyplot as plt
def plot(x, y, title='Data Plot', xlabel='X-axis', ylabel='Y-axis', y_scale=[None, None]):
    plt.figure(figsize=(10, 6))
    
    for k, v in y.items():
        label = k.split('-')[0]
        number = k.split('-')[-1] if '-' in k else 'default'
        color = Label_Color_Mapping.get(label, 'black')
        marker_style = Number_Style_Mapping.get(number, 'o-')
        if isinstance(v, list):
            if len(v) == len(x):
                plt.plot(x, v, marker=marker_style[0], linestyle=marker_style[1:], color=color, label=k)
        else:
            # plot a dashed line
            plt.plot(x, [v]*len(x), linestyle='--', label=k, color=color)
    # set y scale
    if y_scale[0] is not None and y_scale[1] is not None:
        plt.ylim(y_scale[0], y_scale[1])
    elif y_scale[0] is not None:
        plt.ylim(y_scale[0], plt.ylim()[1])
    elif y_scale[1] is not None:
        plt.ylim(plt.ylim()[0], y_scale[1])
    plt.xticks(x, x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'logs-png/{title}.png')
    plt.close()    
    
# plot 1: improvement vs difficulty
x_ticks = ['L1', 'L2', 'L3']
turn_data = {
    'Optimal': [11.625, 16.88, 17.25],
    'Think Fast': [15.875, 38.2, 57.54],
    'Parallel-4K': [13, 33.5, 38.87],
    # 'Parallel-8K': [None, 37.25, 43.25],
    # 'Parallel-16K':[None, 35.5, 26.62],
    # 'Parallel-32K': [12, 24.2, 25.6],
    'Think Slow-8K': [None, 81.25, 60.12],
    # 'Think Slow-16K': [None, 24, 23],
    # 'Think Slow-32K': [11.625, 16.88, 18],
}
reward_data = {
    'Optimal': 1.0,
    'Think Fast': [0.25, -1.79, -5.16],
    'Parallel-4K': [0.25, -1.04, -2.4],
    # 'Parallel-8K': [None, -2.75, -2.5],
    # 'Parallel-16K': [None, -1.62, -0.12],
    # 'Parallel-32K': [0.875, 0.25, 0.25],
    'Think Slow-8K': [None, -49, -18.5],
    # 'Think Slow-16K': [None, -0.12, -0.75],
    # 'Think Slow-32K': [1.0, 1.0, 0.875],
}
plot(x_ticks, turn_data, 'Turn_vs_Difficulty', 'Difficulty Level', 'Avg Turn', y_scale = [None, 60])
plot(x_ticks, reward_data, 'Reward_vs_Difficulty', 'Difficulty Level', 'Avg Reward', y_scale = [-6, None])

# plot 2: perfomance vs budget
x_ticks = ['4k', '8k', '16k', '32k']
turn_data = {
    'Optimal': 17.0625,
    'Think Fast': 48,
    'Parallel': [38, 37, 31.06, 24.9],
    'Think Slow': [None, 70, 23.5, 17.44],
}

reward_data = {
    'Optimal': 1.0,
    'Think Fast': -3.475,
    'Parallel': [-2, -1.8, -0.87, 0.25],
    'Think Slow': [None, -33, -0.44, 0.9375]
}
plot(x_ticks, turn_data, 'Turn_vs_Budget', 'Budget', 'Avg Turn', y_scale = [None, 60])
plot(x_ticks, reward_data, 'Reward_vs_Budget', 'Budget', 'Avg Reward', y_scale = [-6, None])

# plot 3: performance vs budget
x_ticks = ['4k', '8k', '32k']
freeway_data = {
    'Safe': [0.01, 0.65, 0.6]
}
plot(x_ticks, freeway_data, 'Freeway_SingleStep_Budget', 'Budget', 'Avg Reward')

# plot 4: performance vs budget
x_ticks = ['4k', '8k', '32k']
airraid_data = {
    'Optimal': [0.03, 0.58, 0.86]
}
plot(x_ticks, airraid_data, 'AirRaid_SingleStep_Budget', 'Budget', 'Avg Reward')

# plot 5: performance vs budget
x_ticks = ['4k', '8k', '32k']
snake_data = {
    'Safe': [0.56, 0.83, 1]
}
plot(x_ticks, snake_data, 'Snake_SingleStep_Budget', 'Budget', 'Avg Reward')