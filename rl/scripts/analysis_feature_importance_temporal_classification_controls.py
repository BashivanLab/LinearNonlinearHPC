import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.tsa.stattools as smtsa # For acf function
import seaborn as sns
from scipy import stats
import pandas as pd # For creating DataFrame for plotting

sns.set_theme()
sns.set_context("notebook")
sns.set_style("white")

# TODO COPIED FROM analysis_feature_importance.py
def get_trial_phases(row):
    transform_9_alloc_to_5dir = {  # keys = (direction, 9_allocentric_location)
        ('North', 6): 3,  # Corr S
        ('South', 6): 3,  # Corr S
        ('North', 5): 3,  # Center
        ('South', 5): 3,  # Center
        ('North', 4): 3,  # Corr N
        ('South', 4): 3,  # Corr N

        ('South', 9): 5,  # Goal SE
        ('South', 8): 5,  # Goal SW
        ('South', 7): 4,  # Dec S
        ('South', 3): 2,  # Dec N
        ('South', 2): 1,  # Goal NE
        ('South', 1): 1,  # Goal NW

        ('North', 9): 1,  # Goal SE
        ('North', 8): 1,  # Goal SW
        ('North', 7): 2,  # Dec S
        ('North', 3): 4,  # Dec N
        ('North', 2): 5,  # Goal NE
        ('North', 1): 5,  # Goal NW
    }
    current_row = np.array(row)
    dir_index = np.where(current_row[[9, 10]] == 1)[0][0]
    if dir_index == 0:
        direction = 'South'
    else:
        direction = 'North'
    loc_index = np.where(current_row[0:9] == 1)[0][0]
    phase_index = transform_9_alloc_to_5dir[(direction, loc_index + 1)] - 1
    return np.eye(5)[phase_index]

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

temporal_classification_subpath = "figures/feature_importance/23feb25_temporal_classification_mean_v2_p01_distance10/temporal_selectivity_p0.01.pickle"
pickle_in = open('metadata.pickle', "rb")
metadata = pickle.load(pickle_in)
pickle_in.close()
sample_index = 0
fr_index = 1
factors_start_index = 2
n_steps = 30 # This should match the n_steps used in the memory_distance calculation
add_trial_phases = True
max_lags_acf = n_steps  # Max lags for ACF calculation
pvalue_threshold = 0.05 # For the ranksums test, not related to ACF CIs directly here.
experiment_name = args.exp_name

perceptual_color = '#008381' 
near_color = '#ab84a9' 
distant_color = '#a14da0'
temporal_colors = [perceptual_color, near_color, distant_color]

save_fig_folder = f"rl/figures/feature_importance/{experiment_name}"
folder_path = "rl/"
temporal_classification_path = os.path.join(folder_path, temporal_classification_subpath)


if os.path.exists(temporal_classification_path):
    with open(temporal_classification_path, 'rb') as handle:
        temporal_data = pickle.load(handle)
    neuron_names_classified = temporal_data['neuron_ids']
    long_memory_neurons_mask = temporal_data['long_memory_neurons']
    current_trial_memory_neurons_mask = temporal_data['current_trial_memory_neurons']
    perceptual_neurons_mask = temporal_data['perceptual_neurons']
    memory_distance = temporal_data['memory_distance']
else:
    raise FileNotFoundError(f"Temporal classification file not found at: {temporal_classification_path}")

acf_analysis_save_folder = os.path.join(save_fig_folder, "autocorrelation_analysis")
os.makedirs(acf_analysis_save_folder, exist_ok=True)

print(f"Starting ACF analysis for {len(neuron_names_classified)} neurons...")

# For the first plot (proportion with significant ACF at memory distance)
total_perceptual, sig_acf_perceptual = 0, 0
total_near, sig_acf_near = 0, 0
total_distant, sig_acf_distant = 0, 0

# For the second plot (sum of ACF values)
sum_acf_values_perceptual = []
sum_acf_values_near = []
sum_acf_values_distant = []

for i, neuron_id in enumerate(neuron_names_classified):
    if neuron_id not in metadata:
        continue

    md_lag_float = memory_distance[i]
    if np.isnan(md_lag_float):
        continue
    md_lag = int(md_lag_float)

    neuron_activity_full = np.array([e[fr_index] for e in metadata[neuron_id]])

    acf_values = smtsa.acf(neuron_activity_full, nlags=max_lags_acf, fft=True)

        
    # --- Analysis 1: Significance of ACF at memory_distance lag ---
    is_significant_at_md = False
    acf_val_at_md = acf_values[md_lag]
    md_stat_res = stats.ranksums(acf_val_at_md, acf_values[~np.isin(np.arange(len(acf_values)), [0, md_lag])], alternative='greater', nan_policy='omit', axis=None)
    if md_stat_res.pvalue <= pvalue_threshold:
        is_significant_at_md = True

    # --- Analysis 2: Sum of absolute ACF values (excluding lag 0) ---
    current_sum_abs_acf = np.mean(np.abs(acf_values[1:max_lags_acf+1]))

    # Categorize neuron and update counts/lists
    if perceptual_neurons_mask[i]:
        total_perceptual += 1
        if is_significant_at_md:
            sig_acf_perceptual += 1
        sum_acf_values_perceptual.append(current_sum_abs_acf)
    elif current_trial_memory_neurons_mask[i]: # "Near Memory"
        total_near += 1
        if is_significant_at_md:
            sig_acf_near += 1
        sum_acf_values_near.append(current_sum_abs_acf)
    elif long_memory_neurons_mask[i]: # "Distant Memory"
        total_distant += 1
        if is_significant_at_md:
            sig_acf_distant += 1
        sum_acf_values_distant.append(current_sum_abs_acf)


# --- Plot 1: Proportion of Neurons with Significant ACF at Memory Distance Lag ---
categories_plot = ["Perceptual", "Near Memory", "Distant Memory"]
totals_plot = [total_perceptual, total_near, total_distant]
significants_plot = [sig_acf_perceptual, sig_acf_near, sig_acf_distant]

proportions_plot = []
for k_idx in range(len(categories_plot)):
    if totals_plot[k_idx] > 0:
        proportions_plot.append(significants_plot[k_idx] / totals_plot[k_idx])
    else:
        proportions_plot.append(0) 

df_plot1 = pd.DataFrame({
    'Category': categories_plot,
    'ProportionSignificantACF': proportions_plot,
    'CountSignificantACF': significants_plot,
    'TotalInCategory': totals_plot
})

sns.set_theme(context="talk", style='white', font_scale=1.1)
plt.figure(figsize=(8, 6))
bar_plot = sns.barplot(x='Category', y='ProportionSignificantACF', data=df_plot1, palette=temporal_colors)
sns.despine(top=True, right=True)
plt.ylabel("Proportion of neurons with high autocorrelation\nat the most predictive timestep")
plt.ylim(0, max(1.05, df_plot1['ProportionSignificantACF'].max() * 1.1 if df_plot1['ProportionSignificantACF'].max() > 0 else 0.1) ) 

for i_bar, bar in enumerate(bar_plot.patches):
    y_val = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        y_val + 0.01, 
        f"{significants_plot[i_bar]}/{totals_plot[i_bar]}",
        ha='center', 
        va='bottom',
        fontsize=14
    )

plt.tight_layout()
fig1_summary_filename = "summary_acf_significance_at_memory_distance.png"
fig1_summary_path = os.path.join(acf_analysis_save_folder, fig1_summary_filename)
plt.savefig(fig1_summary_path, dpi=300)
plt.savefig(fig1_summary_path.replace(".png", ".svg"), format='svg')
plt.close()
print(f"Plot 1 saved to: {fig1_summary_path}")
print("Data for Plot 1:")
print(df_plot1)


# --- Plot 2: Average Sum of Absolute ACF Values per Category ---
data_for_plot2 = []
for category, sums in zip(categories_plot, [sum_acf_values_perceptual, sum_acf_values_near, sum_acf_values_distant]):
    for val in sums:
        data_for_plot2.append({'Category': category, 'SumAbsACF': val})

df_plot2 = pd.DataFrame(data_for_plot2)
plt.figure(figsize=(8, 6))
# `errorbar='sd'` tells seaborn to calculate and show standard deviation
bar_plot = sns.barplot(x='Category', y='SumAbsACF', data=df_plot2, palette=temporal_colors, errorbar='sd')
sns.despine(top=True, right=True)
# plt.ylabel(f"Mean of Absolute autocorrelation values (Lags 1-{max_lags_acf})")
plt.ylabel(f"Mean autocorrelation across all timesteps (Lags 1-{max_lags_acf})")
plt.title("Overall Neural Activity Autocorrelation Strength")

plt.tight_layout()
fig2_summary_filename = "summary_mean_sum_abs_acf.png"
fig2_summary_path = os.path.join(acf_analysis_save_folder, fig2_summary_filename)
plt.savefig(fig2_summary_path, dpi=300)
plt.savefig(fig2_summary_path.replace(".png", ".svg"), format='svg')
plt.close()
print(f"Plot 2 saved to: {fig2_summary_path}")
print("\nData for Plot 2 (showing first few rows if large):")
print(df_plot2.head())
print("\nAggregated stats for Plot 2:")
print(df_plot2.groupby('Category')['SumAbsACF'].agg(['mean', 'std', 'count']))

perceptual_near_res = stats.ranksums(sum_acf_values_perceptual, sum_acf_values_near, alternative='two-sided', nan_policy='omit', axis=None)
near_distant_res = stats.ranksums(sum_acf_values_near, sum_acf_values_distant, alternative='two-sided', nan_policy='omit', axis=None)
perceptual_distant_res = stats.ranksums(sum_acf_values_perceptual, sum_acf_values_distant, alternative='two-sided', nan_policy='omit', axis=None)
print('pvalues:', perceptual_near_res.pvalue, near_distant_res.pvalue, perceptual_distant_res.pvalue)



print(f"Finished all ACF analyses. Plots saved to: {acf_analysis_save_folder}")