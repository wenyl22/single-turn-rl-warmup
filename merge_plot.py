def merge_plot(plot_list, output_path):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(1, len(plot_list), width_ratios=[1] * len(plot_list))

    for i, plot_path in enumerate(plot_list):
        ax = fig.add_subplot(gs[i])
        img = plt.imread(plot_path)
        ax.imshow(img)
        ax.axis('off')  # Hide the axes

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# plot1_path = 'logs-png/Turn_vs_Budget.png'
# plot2_path = 'logs-png/Reward_vs_Budget.png'
# output_path = 'logs-png/Merged_Turn_Reward_vs_Budget.png'
# merge_plot([plot1_path, plot2_path], output_path)
# plot1_path = 'logs-png/Turn_vs_Difficulty.png'
# plot2_path = 'logs-png/Reward_vs_Difficulty.png'
# output_path = 'logs-png/Merged_Turn_Reward_vs_Difficulty.png'
# merge_plot([plot1_path, plot2_path], output_path)

plot1_path = 'logs-png/Freeway_SingleStep_Budget.png'
plot2_path = 'logs-png/AirRaid_SingleStep_Budget.png'
plot3_path = 'logs-png/Snake_SingleStep_Budget.png'
output_path = 'logs-png/Merged_SingleStep_Budget.png'
merge_plot([plot1_path, plot2_path, plot3_path], output_path)