import os
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np

# Directory containing subfolders with log.csv files
dir_path = 'rl/storage/logcsv'
filter_strings = ['v2_rnn_3_4', 'v2_rnn_3_5', 'v2_rnn_3_6', 'v2_linear_rnn_3_4', 'v2_linear_rnn_3_5', 'v2_linear_rnn_3_6']

column_names = ['update','frames','FPS','duration','perc_optimal_choice','num_choice','perc_incomplete_trials','num_trials',
                'avg_num_trial_per_episode','rreturn_mean','rreturn_std','rreturn_min','rreturn_max','num_frames_mean','num_frames_std',
                'num_frames_min','num_frames_max','entropy','value','policy_loss','value_loss','grad_norm','learning_rate',
                'learning_rate_duplicate','return_mean','return_std','return_min','return_max']
# List to store the data for each subfolder
data_list = []

# Loop through each subfolder in the directory
for subfolder in os.listdir(dir_path):
    subfolder_path = os.path.join(dir_path, subfolder)
    log_file = os.path.join(subfolder_path, 'log.csv')

    # Check if log.csv exists in the subfolder
    if os.path.isfile(log_file) and os.path.getsize(log_file) > 0:

        if filter_strings == [] or any(substr in log_file for substr in filter_strings):
            # Check if need to filter some files and skip files not containing any of the filter_string

            # Load the CSV file into a DataFrame
            # df = pd.read_csv(log_file, )
            df = pd.read_csv(log_file, index_col=None, header=None, names=column_names)

            # Get the last row of the CSV file
            last_row = df.iloc[-1]

            # Extract the specified variables
            update = last_row['update']
            perc_optimal_choice = last_row['perc_optimal_choice']
            num_choice = last_row['num_choice']
            perc_incomplete_trials = last_row['perc_incomplete_trials']
            avg_num_trial_per_episode = last_row['avg_num_trial_per_episode']
            rreturn_mean = last_row['rreturn_mean']

            # Append the extracted data along with the subfolder name
            data_list.append({
                'subfolder': subfolder,
                'update': int(update),
                'perc_optimal_choice': np.round(float(perc_optimal_choice),2),
                'num_choice': int(num_choice),
                'perc_incomplete_trials': float(perc_incomplete_trials),
                'avg_num_trial_per_episode': float(avg_num_trial_per_episode),
                'rreturn_mean': float(rreturn_mean)
            })

# Convert the data list to a DataFrame for easier manipulation
results_df = pd.DataFrame(data_list)
results_df = results_df.sort_values(by='subfolder', ascending=True)

# Function to plot a variable across all subfolders
def plot_variable(df, variable_name, ylabel):
    # Filter not completely trained models
    df_filtered = df[df['update'] >= 344 * 50]
    plt.figure(figsize=(10, 6))
    plt.bar(df_filtered['subfolder'], df_filtered[variable_name]) # color='skyblue'
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.xlabel('Subfolder')
    plt.ylabel(ylabel)
    # plt.title(f'{ylabel} across Subfolders')
    plt.tight_layout()
    plt.show()


# Plot each variable individually
plot_variable(results_df, 'update', 'Update')
plot_variable(results_df, 'perc_optimal_choice', 'Percentage of Optimal Choices')
plot_variable(results_df, 'num_choice', 'Number of Choices')
plot_variable(results_df, 'perc_incomplete_trials', 'Percentage of Incomplete Trials')
plot_variable(results_df, 'avg_num_trial_per_episode', 'Average Number of Trials per Episode')
plot_variable(results_df, 'rreturn_mean', 'Return Mean')

df_filtered = results_df[results_df['update'] >= 344 * 50]
df_filtered = df_filtered[df_filtered['perc_optimal_choice'] >= 75]
df_filtered['subfolder']