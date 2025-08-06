import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model-type", default=None,
                    help="type of model to use (default from model.py)")
parser.add_argument("--exp-name", default=None,
                    help="Name of the experiment to identify the output files")
args = parser.parse_args()

sns.set_theme()
sns.set_context("notebook")
sns.set_style("white")

experiment_name = args.exp_name

path_trial_shuffle = "rl/figures/feature_importance/28fmay25_shuffled_temporal_classification_mean_v2_p01_distance10/row_wise_profile_trial_shuffle.pickle"
path_original = "rl/figures/feature_importance/14sept24_feature_importance/row_wise_profile.pickle"
temporal_trial_shuffled_path = "figures/feature_importance/28fmay25_shuffled_temporal_classification_mean_v2_p01_distance10/temporal_selectivity_p0.01_trial_shuffled.pickle"
temporal_path = "figures/feature_importance/23feb25_temporal_classification_mean_v2_p01_distance10/temporal_selectivity_p0.01.pickle"
number_step = 30
save_fig = True
use_moving_average = True
plot_individual_profiles = False
plot_histogram_memory_distance = False
best_np_comparison = True

save_fig_folder = f"rl/figures/feature_importance/{experiment_name}"
folder_path = "rl/"


os.makedirs(f"{save_fig_folder}", exist_ok=True)


with open(path_original, 'rb') as handle:
    row_wise_corr_per_neuron = pickle.load(handle)
with open(path_trial_shuffle, 'rb') as handle:
    row_wise_corr_per_neuron_trial_shuffle = pickle.load(handle)

def plot_distribution_hist(data, x_label, y_label, legend_labels, filename, discret_hist=True, colors=None, title=None,
                           bins='auto', yticks=None, xticks=None, add_mean=True):
    if colors is None:
        colors = sns.color_palette('deep')

    # Calculate common bins if `bins='auto'` is specified
    if bins == 'auto':
        # Combine data to determine common bins
        cleaned_data = [np.array(d)[~np.isnan(d)] for d in data]
        all_data = np.concatenate(cleaned_data)
        bins = np.histogram_bin_edges(all_data, bins='auto')
    if isinstance(bins, int):
        cleaned_data = [np.array(d)[~np.isnan(d)] for d in data]
        all_data = np.concatenate(cleaned_data)
        bins = np.histogram_bin_edges(all_data, bins=bins)

    # Set plot transparency and add different line styles
    for i, d in enumerate(data):
        linestyle = '-'
        linewidth = 4
        sns.histplot(d, discrete=discret_hist, stat='proportion', element='step', fill=False, #alpha=0.6,
                     common_norm=False, legend=False, color=colors[i], bins=bins,
                     linewidth=linewidth, linestyle=linestyle)


    # Configure plot
    if legend_labels is not None:
        plt.legend(labels=legend_labels, loc='upper right', frameon=False,)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if title is not None:
        # plt.title(title, fontsize=14, weight='bold')
        plt.title(title, fontsize=18)
    if yticks is not None:
        plt.yticks(yticks)
    if xticks is not None:
        plt.xticks(xticks)
    sns.despine(top=True, right=True)

    if add_mean:
        # Plot mean as vertical line
        ymin, ymax = plt.ylim()
        for i, d in enumerate(data):
            mean_value = np.nanmean(d)  # Ignore NaNs when computing mean
            plt.vlines(mean_value, ymin, ymax * 0.9, color=colors[i], linestyle='dashed', linewidth=2)

    # Save or display plot
    if save_fig:
        os.makedirs(f"{save_fig_folder}/", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', dpi=900)
        os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/svg/{filename}.svg", bbox_inches='tight', dpi=900, format='svg')
        plt.clf()
    else:
        plt.show()

def moving_average(data, window_size=3):
    pad_size = window_size // 2
    if len(data) < window_size:
        return np.convolve(data, np.ones(len(data)) / len(data), mode='valid')
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    smoothed = np.concatenate(([np.mean(data[:pad_size + 1])], smoothed,
                               [np.mean(data[-pad_size - 1:])]))  # Average the ends with partial moving window
    return smoothed



if plot_individual_profiles:
    for neuron_id in row_wise_corr_per_neuron.keys():
        sns.set_theme(context="talk", style='white', font_scale=1.2)
        sns.set_theme(context="talk", style='white', font_scale=2)
        mean = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0)
        std = np.nanstd(row_wise_corr_per_neuron[neuron_id], axis=0)

        if use_moving_average:
            smoothed_data = np.array([moving_average(data) for data in row_wise_corr_per_neuron[neuron_id]])
            mean = np.nanmean(smoothed_data, axis=0)
            std = np.nanstd(smoothed_data, axis=0)

            smoothed_data_trial_shuffle = np.array([moving_average(data) for data in row_wise_corr_per_neuron_trial_shuffle[neuron_id]])
            mean_trial_shuffle = np.nanmean(smoothed_data_trial_shuffle, axis=0)
            std_trial_shuffle = np.nanstd(smoothed_data_trial_shuffle, axis=0)



        if isinstance(number_step, tuple):
            time_range = [i for i in range(number_step[1], -number_step[0], -1)]
        else:
            time_range = [i for i in range(number_step)]

        # Set background colors
        perceptual_color = '#008381'
        near_color = '#ab84a9'
        distant_color = '#a14da0'
        plt.axvspan(-0.5, 0.5, color=perceptual_color, alpha=0.7)
        plt.axvspan(0.51, 10.5, color=near_color, alpha=0.7)
        plt.axvspan(10.51, 29.5, color=distant_color, alpha=0.7)

        plt.plot(time_range, mean, color='black', linewidth=3)
        plt.fill_between(time_range, mean - std, mean + std, color='lightgray', alpha=0.2)

        plt.plot(time_range, mean_trial_shuffle, color='black', linewidth=3, linestyle='--')
        plt.fill_between(time_range, mean_trial_shuffle - std_trial_shuffle, mean_trial_shuffle + std_trial_shuffle, color='lightgray', alpha=0.2)

        plt.gca().invert_xaxis()
        plt.ylabel('Neural Predictivity')
        plt.xlabel('Time steps')
        neuron_name = neuron_id[0] + neuron_id[5:].replace('_', '.')
        plt.title(f'{neuron_name}')
        sns.despine()
        plt.tight_layout()
        if save_fig:
            os.makedirs(f"{save_fig_folder}/row-wise-profiles/", exist_ok=True)
            fig_path = f"{save_fig_folder}/row-wise-profiles/"
            plt.savefig(fig_path + neuron_id, bbox_inches='tight', dpi=900)
            os.makedirs(f"{fig_path}svg", exist_ok=True)
            plt.savefig(f"{fig_path}svg/{neuron_id}.svg", bbox_inches='tight', dpi=900, format='svg')
            plt.clf()
            plt.close()
        else:
            plt.show()



if plot_histogram_memory_distance:
    with open(temporal_path, 'rb') as handle:
        temporal_data = pickle.load(handle)
    with open(temporal_trial_shuffled_path, 'rb') as handle:
        temporal_trial_shuffled_data = pickle.load(handle)

    data = [temporal_data['memory_distance'], temporal_trial_shuffled_data['memory_distance']]
    plot_distribution_hist(data, x_label="Memory distance",
                           y_label="Proportion of the population",
                           discret_hist=False,
                           bins=(0,1.1, 5.1, 10.1, 15.1, 20.1, 25.1, 30),
                           # bins=(0,1.1, 10.1, 30),
                           # bins=7,
                           legend_labels=["", "Within trial shuffled"],
                           filename='shift_mem_distance_hist',
                           colors=["black", "grey"], add_mean=True)

if best_np_comparison:
    best_np = []
    best_np_trial_shuffled = []
    for neuron_id in row_wise_corr_per_neuron.keys():
        best_np.append(np.nanmax(row_wise_corr_per_neuron[neuron_id], axis=1))
        best_np_trial_shuffled.append(np.nanmax(row_wise_corr_per_neuron_trial_shuffle[neuron_id], axis=1))
    best_np = np.array(best_np)
    best_np_trial_shuffled = np.array(best_np_trial_shuffled)

    mean_original = np.nanmean(best_np)
    std_original = np.nanstd(np.nanmean(best_np,axis=0))

    mean_shuffled = np.nanmean(best_np_trial_shuffled)
    std_shuffled = np.nanstd(np.nanmean(best_np_trial_shuffled,axis=0))

    individual_dots = np.nanmean(best_np, axis=1)
    individual_dots_shuffled = np.nanmean(best_np_trial_shuffled, axis=1)
    # --- Prepare data for stripplot (long format) ---
    # We need a DataFrame with columns like: 'value', 'category'
    data_for_dots = []
    for val in individual_dots:
        if not np.isnan(val):  # Only include non-NaN values
            data_for_dots.append({'value': val, 'category': 'Original'})
    for val in individual_dots_shuffled:
        if not np.isnan(val):  # Only include non-NaN values
            data_for_dots.append({'value': val, 'category': 'Within trial shuffled'})
    df_dots = pd.DataFrame(data_for_dots)
    # --- End data preparation for stripplot ---

    # Data for plotting
    labels = ['Original', 'Within trial shuffled']
    means = [mean_original, mean_shuffled]
    errors = [std_original, std_shuffled]  # Using std as error

    x_pos = np.arange(len(labels))

    # Bar plot
    sns.set_theme(context="talk", style='white', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(x_pos, means, yerr=errors, align='center', ecolor='black', capsize=10,
                  color=['#4F5669', '#4F5669'])

    # --- Add connecting lines for individual data points ---
    # These lines connect the (x_center_of_bar, actual_value_for_neuron_i) points
    # individual_dots[i] corresponds to individual_dots_shuffled[i] for the same neuron
    # because common_neuron_ids was sorted and iterated upon.
    num_neurons = len(individual_dots)
    for i in range(num_neurons):
        val_orig = individual_dots[i]
        val_shuf = individual_dots_shuffled[i]

        # Only draw a line if both points are valid (not NaN)
        if not np.isnan(val_orig) and not np.isnan(val_shuf):
            ax.plot([x_pos[0], x_pos[1]], [val_orig, val_shuf],
                    color='dimgray',  # A visible but not overpowering color for lines
                    linestyle='-',
                    linewidth=0.8,  # Thinner lines
                    alpha=0.5,  # Semi-transparent
                    zorder=2)  # Lines on top of bars (zorder=1), below dots (zorder=3)
    # --- End connecting lines ---

    # --- Add individual data points using stripplot ---
    sns.stripplot(
        x='category',
        y='value',
        data=df_dots,
        order=labels,  # Ensures the order matches the bars
        color='black',  # Color of the dots
        jitter=0.15,  # Amount of horizontal spread (0 for no jitter, True for default)
        size=6,  # Size of the dots
        alpha=0.7,  # Transparency of dots
        ax=ax  # Plot on the same axes as the bars
    )

    ax.set_ylabel('Neural predictivity\nfrom most predictive time step')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    if save_fig:
        os.makedirs(f"{save_fig_folder}/", exist_ok=True)
        fig_path = f"{save_fig_folder}/np_bar_plot_trial_shuffled.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=900)
        plt.savefig(fig_path[:-3] + 'svg', bbox_inches='tight', format='svg')
        plt.clf()
        plt.close()
    else:
        plt.show()
    plt.show()

    #########
    # Statistical test
    #########
    mean_best_np = np.nanmean(best_np, axis=1)
    mean_best_np_trial_shuffled = np.nanmean(best_np_trial_shuffled, axis=1)
    wilcoxon_stat, wilcoxon_p_value = stats.wilcoxon(
        mean_best_np,
        mean_best_np_trial_shuffled,
        alternative='greater', # one-tailed test
        nan_policy='omit',
    )

    print(f"\nWilcoxon signed-rank test:")
    print(f"Statistic: {wilcoxon_stat:.4f}, p-value: {wilcoxon_p_value}")
