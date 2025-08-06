import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

save_fig_folder = f"rl/figures/learning_curves"

roarn_color = '#D7263D'
linear_color = '#007A3D'

# Directory containing subfolders with log.csv files
dir_path = 'rl/storage/logcsv'
filter_rnn = ['v2_rnn_3_4', 'v2_rnn_3_5', 'v2_rnn_3_6']
filter_linear = ['v2_linear_rnn_3_4', 'v2_linear_rnn_3_5', 'v2_linear_rnn_3_6']

column_names = ['update', 'frames', 'FPS', 'duration', 'perc_optimal_choice', 'num_choice', 'perc_incomplete_trials', 'num_trials',
                'avg_num_trial_per_episode', 'rreturn_mean', 'rreturn_std', 'rreturn_min', 'rreturn_max', 'num_frames_mean', 'num_frames_std',
                'num_frames_min', 'num_frames_max', 'entropy', 'value', 'policy_loss', 'value_loss', 'grad_norm', 'learning_rate', 'return_mean', 'return_std', 'return_min', 'return_max']

# Lists to store dataframes
rnn_dfs = []
linear_dfs = []

# Loop through each subfolder in the directory
for subfolder in os.listdir(dir_path):
    subfolder_path = os.path.join(dir_path, subfolder)
    log_file = os.path.join(subfolder_path, 'log.csv')

    # Check if log.csv exists and is not empty
    if os.path.isfile(log_file) and os.path.getsize(log_file) > 0:
        # Determine if the file belongs to the RNN or Linear group
        if any(substr in log_file for substr in filter_rnn):
            df = pd.read_csv(log_file, index_col=None, header=0, names=column_names)
            rnn_dfs.append(df)
        elif any(substr in log_file for substr in filter_linear):
            df = pd.read_csv(log_file, index_col=None, header=0, names=column_names)
            linear_dfs.append(df)

# Ensure we have data before proceeding
if not rnn_dfs or not linear_dfs:
    raise ValueError("One or both groups have no valid files.")

# Convert lists of DataFrames into a single DataFrame per group
rnn_df = pd.concat(rnn_dfs).groupby('update').mean()
rnn_std = pd.concat(rnn_dfs).groupby('update').std()

linear_df = pd.concat(linear_dfs).groupby('update').mean()
linear_std = pd.concat(linear_dfs).groupby('update').std()

# Filter updates <= 17200
rnn_df = rnn_df[rnn_df.index <= 17200]
rnn_std = rnn_std[rnn_std.index <= 17200]

linear_df = linear_df[linear_df.index <= 17200]
linear_std = linear_std[linear_std.index <= 17200]

sns.set_theme(context="talk", style='white', font_scale=1.2)
metrics = ['perc_optimal_choice', 'num_choice', 'num_trials', 'rreturn_mean']
metric_labels = ['Make optimal decision (%)', 'Number of completed trials', 'Number of completed trials', 'rreturn_mean']
for i in range(len(metrics)):
    # Plot the data
    plt.figure(figsize=(10, 6))

    plt.plot(rnn_df.index, rnn_df[metrics[i]], label="ROARN", color=roarn_color)
    plt.fill_between(rnn_df.index, rnn_df[metrics[i]] - rnn_std[metrics[i]], rnn_df[metrics[i]] + rnn_std[metrics[i]], color=roarn_color, alpha=0.3)

    plt.plot(linear_df.index, linear_df[metrics[i]], label="ROARN w/ LR", color=linear_color)
    plt.fill_between(linear_df.index, linear_df[metrics[i]] - linear_std[metrics[i]], linear_df[metrics[i]] + linear_std[metrics[i]], color=linear_color, alpha=0.3)

    plt.xticks([2000, 8000, 17000])
    plt.xlabel("Training updates")
    plt.ylabel(metric_labels[i])
    plt.legend()
    sns.despine()
    plt.tight_layout()

    filename = metrics[i]
    os.makedirs(f"{save_fig_folder}/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/svg/{filename}.svg", bbox_inches='tight', dpi=900,
                format='svg')
    plt.clf()
    plt.close()

    # ZOOMED-IN PLOT
    plt.figure(figsize=(10, 6))
    zoomed_start = 2000
    zoomed_end = 8000

    # Filter for zoomed-in range
    zoom_rnn = rnn_df[(rnn_df.index >= zoomed_start) & (rnn_df.index <= zoomed_end)]
    zoom_rnn_std = rnn_std[(rnn_std.index >= zoomed_start) & (rnn_std.index <= zoomed_end)]

    zoom_linear = linear_df[(linear_df.index >= zoomed_start) & (linear_df.index <= zoomed_end)]
    zoom_linear_std = linear_std[(linear_std.index >= zoomed_start) & (linear_std.index <= zoomed_end)]

    plt.plot(zoom_rnn.index, zoom_rnn[metrics[i]], label="ROARN (Zoomed)", color=roarn_color)
    plt.fill_between(zoom_rnn.index, zoom_rnn[metrics[i]] - zoom_rnn_std[metrics[i]],
                     zoom_rnn[metrics[i]] + zoom_rnn_std[metrics[i]], color=roarn_color, alpha=0.3)

    plt.plot(zoom_linear.index, zoom_linear[metrics[i]], label="ROARN w/ LR (Zoomed)", color=linear_color)
    plt.fill_between(zoom_linear.index, zoom_linear[metrics[i]] - zoom_linear_std[metrics[i]],
                     zoom_linear[metrics[i]] + zoom_linear_std[metrics[i]], color=linear_color, alpha=0.3)

    plt.xticks([2000, 8000])
    plt.xlabel("Training updates")
    plt.ylabel(metric_labels[i])
    plt.tight_layout()

    filename = metrics[i]
    os.makedirs(f"{save_fig_folder}/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/{filename}_zoomed", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/svg/{filename}_zoomed.svg", bbox_inches='tight', dpi=900,
                format='svg')
    plt.clf()
    plt.close()