import os
import argparse
import ast
from random import shuffle
from random import sample
from scripts.create_metadata_matrix import align_metadata_and_activation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import linear_model
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr, pointbiserialr
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

aligning_with_activations = False
add_trial_phases = True
save_fig = True
sns.set_theme()
sns.set_context("notebook")
sns.set_style("white")

experiment_name = args.exp_name
if aligning_with_activations:
    experiment_name += "_aligned_w_activations"

h5_file_path = f'activations_{args.model}_{args.env}.h5'
save_fig_folder = f"rl/figures/feature_importance/{experiment_name}"
folder_path = "rl/"

os.makedirs(f"{save_fig_folder}", exist_ok=True)

if args.model_type is not None and 'epn' in args.model_type:
    layer_name = 'max_layer'
else:
    layer_name = 'cell_state'

# Load activations
if aligning_with_activations:
    hf = h5py.File(h5_file_path, 'r')
    activations = hf[f"aligned_activations_{layer_name}"]
pickle_in = open('metadata.pickle', "rb")
metadata = pickle.load(pickle_in)
pickle_in.close()
sample_index = 0
fr_index = 1
factors_start_index = 2
n_steps = 30

do_row_wise = True
load_previous_row_wise = True
plot_row_wise = False


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

# TODO COPY-PASTED from spatial_neurons.py
from scipy.stats import sem
def grouped_bar_plot(data, group_names, bin_names, error_type, xlabel, ylabel, filename, folder_name=None,
                     keep_legend=False, averaging_axis=1):
    # Data: list with shape (nb_group, number of neurons in this group, number of bins)
    # if averaging_axis=1, average across dimension 1 of data (or across dimension averaging_axis-1 of data[i])
    # if averaging_axis=2, data shape = (nb_group, number of bins, number of neurons in this combination of group and bin)
    # If error_type is None, Doesn't average the second dimension of Data. (so works with 2D data)
    if error_type is not None:
        # average across the n-dimension of data for each group, thus across averaging_axis-1 of data[i]
        averaging_axis = averaging_axis - 1
        mean_per_bin = []
        median_per_bin = []
        std_per_bin = []
        sem_per_bin = []
        if averaging_axis == 0:
            for i in range(len(group_names)):
                mean_per_bin.append(np.mean(data[i], axis=averaging_axis))
                median_per_bin.append(np.median(data[i], axis=averaging_axis))
                std_per_bin.append(np.std(data[i], axis=averaging_axis))
                sem_per_bin.append(sem(data[i], axis=averaging_axis))
        else:
            for i in range(len(group_names)):
                mean_per_bin.append([])
                median_per_bin.append([])
                std_per_bin.append([])
                sem_per_bin.append([])
                for j in range(len(bin_names)):
                    mean_per_bin[i].append(np.mean(data[i][j], axis=0))
                    median_per_bin[i].append(np.median(data[i][j], axis=0))
                    std_per_bin[i].append(np.std(data[i][j], axis=0))
                    sem_per_bin[i].append(sem(data[i][j], axis=0))

        df = {
            'mean': np.array(mean_per_bin).flatten(),
            'median': np.array(median_per_bin).flatten(),
            'std': np.array(std_per_bin).flatten(),
            'sem': np.array(sem_per_bin).flatten(),
            'neuron_type': np.tile(group_names, (len(bin_names), 1)).T.flatten(),
            'bin_name': np.tile(bin_names, (len(group_names), 1)).flatten()
        }
    else:
        df = {
            'data': np.array(data).flatten(),
            'neuron_type': np.tile(group_names, (len(bin_names), 1)).T.flatten(),
            'bin_name': np.tile(bin_names, (len(group_names), 1)).flatten()
        }
    df = pd.DataFrame(df)
    for m in ['mean', 'median']:
        if error_type is None:
            m = 'data'
        ax = sns.barplot(data=df, x='bin_name', y=m, hue='neuron_type',
                         palette=sns.set_palette(sns.color_palette("deep")))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        # plt.ylim(0.0, 1.0)
        if not keep_legend:
            sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, -0.35), ncol=4, title=None, frameon=False)
            # if save_legend:
            #     export_legend(ax, filename, folder_name)
            ax.get_legend().remove()
        else:
            ax.legend().set_title('')
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        if error_type is not None:
            plt.errorbar(x=x_coords, y=y_coords, yerr=df[error_type], fmt="none", c="k", capsize=3)  # , elinewidth=0.5
        plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        if save_fig:
            if folder_name is not None:
                os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}_{m}", bbox_inches='tight', dpi=900)
            else:
                plt.savefig(f"{save_fig_folder}/{filename}_{m}", bbox_inches='tight', dpi=900)
                # plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', format='svg')
            plt.clf()
            plt.close()
        else:
            plt.show()



def get_meta_dataset(neuron_metadata, firing_index, factors_start_index, add_multiplicative_factors=False,
                     delete_n_steps_overlap=False, add_trial_phases=False, n_steps=1):
    """
    Give the input matrix for the regression
    Selection only columns in metadata that are features (i.e. not sample_id and firing rates)
    For each input point, concatenate the last n_steps to create one sample.
    """
    X_metadata = []
    firing_rates = []
    i = 0
    if isinstance(n_steps, tuple):
        future_steps = n_steps[1]
        n_steps = n_steps[0]
    else:
        future_steps = 0
    while i in range(len(neuron_metadata)-future_steps):
        if (i - n_steps < 0) or (neuron_metadata[i][0][0] != neuron_metadata[i - n_steps][0][0]):
            # Not enough previous steps available (in the same session)
            i += 1
            continue

        # Get all future steps and the current time steps
        row = None
        for step in range(future_steps, -1, -1):
            # concatenate all future steps to one sample
            if row is None:
                row = neuron_metadata[i + step][factors_start_index:]
            else:
                row.extend(neuron_metadata[i + step][factors_start_index:])
            if add_trial_phases:
                row.extend(get_trial_phases(neuron_metadata[i + step][factors_start_index:]))

        if row is None:
            # Remove sampled_id and activations
            row = neuron_metadata[i][factors_start_index:]
            if add_trial_phases:
                row.extend(get_trial_phases(row))

            if add_multiplicative_factors:
                # Add all multiplicative combinations for each factors (e.g. f1*f2, f1*f3, ..., f2*f3, ...)
                from itertools import combinations
                all_factor_index = np.arange(len(row))
                for (f1, f2) in combinations(all_factor_index, 2):
                    row.append(row[f1] * row[f2])

        for step in range(1, n_steps):
            # concatenate n-1 previous steps to the sample
            row.extend(neuron_metadata[i - step][factors_start_index:])
            if add_trial_phases:
                row.extend(get_trial_phases(neuron_metadata[i - step][factors_start_index:]))
        X_metadata.append(row)
        firing_rates.append(neuron_metadata[i][firing_index])
        if delete_n_steps_overlap:
            i = i + n_steps
        else:
            i += 1
    X_metadata = np.array(X_metadata)
    firing_rates = np.array(firing_rates)
    return X_metadata, firing_rates



feature_names = ['North-West Arm', 'North-East Arm', 'North Branching', 'North Corridor', 'Corridor Center', 'South Corridor', 'South Branching',
                 'South-West Arm', 'South-East Arm',
                 'South', 'North', 'Steel context', 'Wood context',
                 # 'West color best wood', 'West color middle', 'West color worst wood',
                 # 'East color best wood', 'East color middle', 'East color worst wood',
                 'West object 1', 'West object 2', 'West object 3',
                 'East object 1', 'East object 2', 'East object 3',
                 'Selected object 1', 'Selected object 2', 'Selected object 3',
                 'Worst reward', 'Middle reward', 'Best reward',
                 "Incorrect choice", "Correct choice"]
if add_trial_phases:
    feature_names.extend(
        ['Trial start', 'Precontext', 'Context appearance', 'Object appearance', 'Object approach (Decision)'])


def get_regression_dict(filename, regression_name):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]

    result_per_neuron = {}
    for line in lines:
        regressor = line[line.find("Regressor: ")+len("Regressor: "):line.find(", Best_result:")]
        if regressor != regression_name:
            continue
        neuron_id = line[8:line.find(", Regressor: ")]
        if 'Spearmanr' in line:
            corr = float(line[line.find("Pearsonr: ")+len("Pearsonr: "):line.find(", Spearmanr:")])
        else:
            corr = float(line[line.find("Pearsonr: ")+len("Pearsonr: "):])
        result_per_neuron[neuron_id] = corr
    return result_per_neuron

def get_regression_model_specific_performance(reg_type, filenames, subselection_neurons=None):
    scores = []
    for filename in filenames:
        result_per_neuron = get_regression_dict(filename, reg_type)
        model_score = []
        if subselection_neurons is None:
            subselection_neurons = result_per_neuron.keys()
        for neuron in subselection_neurons:
            # Select the correlation score obtained with regression model 'reg_type'
            model_score.append(result_per_neuron[neuron])
        scores.append(model_score)
    return scores

####################################################################################
# Full matrix feature importance
# For different n_steps
# For each neuron
# 1. Load best regression hyperparameters
# 2. Fit regressor
# 3. Plot weight across matrix representing each features (like in element-wise)
####################################################################################

def get_regression_params(lines, wanted_neuron, regression_model):
    for line in lines:
        neuron = line[8:line.find(", Regressor: ")]
        if wanted_neuron == neuron:
            reg_model = line[line.find("Regressor: ") + len("Regressor: "):line.find(", Best_result:")]
            if reg_model == regression_model:
                params_string = line[line.find("Param: ") + len("Param: "):line.find(", Runtime:")]
                params = ast.literal_eval(params_string)
                return params

# TODO copy-pasted from regression.py
def timeseries_crosval_predict(X, y, reg_model, nb_folds=10):
    """
    X: training samples
    y: labels
    reg_model: regression model to use
    nb_folds:Number of folds. If 10, separate the data into 10 chunks, train on 9 and keep one (ie 10% of the data)
        for evaluation. Change which one is left out, retrain, and test and new eval samples. Repeat until every samples
        is predicted
    Return:
        predicted_ff: prediction for every samples obtained by doing cross-val of every timeseries chunk,
        reg_model: regression model fitted on the last chunks
    """
    kfold_weights = []
    ind_list = [i for i in range(len(y))]
    fold_size = int(len(y) / (3 * nb_folds))
    set_seperated_in_chunk = [ind_list[i:i + fold_size] for i in range(0, len(ind_list), fold_size)]
    left_over_inds = []
    nb_extra_chunks = len(set_seperated_in_chunk) - 3 * nb_folds
    if nb_extra_chunks > 0:
        # indices that didn't fit in the 3*nb_fold equal sized folds are distributed into the folds
        left_over_inds = set_seperated_in_chunk[-nb_extra_chunks:]
        set_seperated_in_chunk = set_seperated_in_chunk[:-nb_extra_chunks]
        shuffle(left_over_inds)
    set_seperated_in_chunk = np.array(set_seperated_in_chunk)
    left_over_per_fold = [[] for _ in range(nb_folds)]
    looping_i = 0
    while len(left_over_inds) != 0:
        left_over_per_fold[looping_i].append(left_over_inds.pop())
        looping_i = (looping_i + 1) % nb_folds
    indices = list(range(nb_folds * 3))  # all indices that are not left_over
    # Get all index separated per third, and shuffle them
    test_chunk_indexes = [sample(indices[i:i + nb_folds], len(indices[i:i + nb_folds])) for i in
                          range(0, len(indices), nb_folds)]
    predicted_ff = [0 for _ in y]
    for _ in range(nb_folds):
        test_ind = set_seperated_in_chunk[[e.pop() for e in test_chunk_indexes]].flatten()
        test_ind = np.append(test_ind, np.array(left_over_per_fold.pop(),
                                                dtype=np.int64))  # Add some left_over_inds in each folds until they are all tested/predicted
        train_ind = [ind for ind in ind_list if ind not in test_ind]
        shuffle(train_ind)
        reg_model.fit(np.array(X)[train_ind, :], np.array(y)[train_ind])
        if isinstance(reg_model, Pipeline):
            kfold_weights.append(reg_model['reg'].coef_)
        else:
            kfold_weights.append(reg_model.coef_)
        shuffled_predicted_ff = reg_model.predict(np.array(X)[test_ind, :])
        for j in range(len(test_ind)):
            predicted_ff[test_ind[j]] = shuffled_predicted_ff[j]
    return predicted_ff, reg_model, kfold_weights



if do_row_wise:
    ####################################################################################
    # For each neuron
    # 1. Load best regression hyperparameters for 1 step
    # 2. For row_index in range(0,30)
    #   2.a) Fit regressor on row_index
    #   2.b) Get correlation score for that row
    # 3. Plot x = row index, y = correlation
    ####################################################################################
    if load_previous_row_wise:
        path = "rl/figures/feature_importance/14sept24_feature_importance/row_wise_profile.pickle"
        number_step = 30
        with open(path, 'rb') as handle:
            # row_wise_corr_per_neuron = pickle.load(handle)
            row_wise_corr_per_neuron = pickle.load(handle)

    else:
        # number_step = (30, 30)  # Number of past time-steps and future time-steps
        number_step = 30

        regression_file = "reg_output/memory-cc-v2_rnn_3_4-Ideal_obs_k10/344.txt"
        filename = folder_path + regression_file

        regression_files = [
            "reg_output/memory-cc-v2_rnn_3_4-Ideal_obs_k10/344.txt",
            "reg_output/memory-cc-v2_rnn_3_5-Ideal_obs_k10/343.txt",
            "reg_output/memory-cc-v2_rnn_3_6-Ideal_obs_k10/383.txt",
        ]
        row_wise_corr_per_neuron = {}

        for regression_file in regression_files:
            filename = folder_path + regression_file
            with open(filename) as file:
                lines = [line.rstrip() for line in file]
            previous_gridsearch_results = lines
            reg_name = 'LinearSVR'
            reg_type = LinearSVR
            best_parameters = get_regression_params(previous_gridsearch_results, list(metadata.keys())[0], regression_model=reg_name)

            # row_wise_corr_per_neuron = {}
            reg = reg_type()
            normalize_input = False
            if not normalize_input:
                reg_model = reg
            else:
                reg_model = Pipeline([('scaler', StandardScaler()), ('reg', reg)])
            nb_seeds = 3
            for s in range(nb_seeds):
                for neuron_id in metadata.keys():
                    if neuron_id not in row_wise_corr_per_neuron:
                        row_wise_corr_per_neuron[neuron_id] = []
                    if aligning_with_activations:
                        neuron_metadata, activations = align_metadata_and_activation(metadata, activations, neuron_id)
                    else:
                        neuron_metadata = metadata[neuron_id]
                    X_metadata, firing_rates = get_meta_dataset(neuron_metadata, fr_index, factors_start_index,
                                                                add_trial_phases=add_trial_phases, n_steps=number_step)
                    reg_model.set_params(**best_parameters)

                    correlation_per_row = []
                    if isinstance(number_step, tuple):
                        nb_steps = np.sum(number_step)
                    else:
                        nb_steps = number_step
                    nb_feature_per_row = int(X_metadata.shape[1] / nb_steps)
                    for row_index in range(nb_steps):
                        row_i = X_metadata[:, nb_feature_per_row * row_index:nb_feature_per_row * (row_index + 1)]
                        predicted_ff, reg_model, kfold_weights = timeseries_crosval_predict(row_i, firing_rates, reg_model)
                        corr, pvalue = pearsonr(predicted_ff, firing_rates)
                        correlation_per_row.append(corr)

                    row_wise_corr_per_neuron[neuron_id].append(np.array(correlation_per_row))

        with open(f"{save_fig_folder}/row_wise_profile.pickle", 'wb') as handle:
            pickle.dump(row_wise_corr_per_neuron, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('.')


    def moving_average(data, window_size=3):
        pad_size = window_size // 2
        if len(data) < window_size:
            return np.convolve(data, np.ones(len(data)) / len(data), mode='valid')
        smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        smoothed = np.concatenate(([np.mean(data[:pad_size + 1])], smoothed,
                                   [np.mean(data[-pad_size - 1:])]))  # Average the ends with partial moving window
        return smoothed

    if plot_row_wise:
        use_moving_average = True
        for neuron_id in row_wise_corr_per_neuron.keys():
            sns.set_theme(context="talk", style='white', font_scale=1.2)
            mean = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0)
            std = np.nanstd(row_wise_corr_per_neuron[neuron_id], axis=0)

            if use_moving_average:
                # Apply moving average
                # mean = moving_average(mean, window_size=3)
                # std = moving_average(std, window_size=3)
                # time_range = time_range[:len(mean)]  # Adjust time range if convolution changes the size, e.g. mode='valid'
                smoothed_data = np.array([moving_average(data) for data in row_wise_corr_per_neuron[neuron_id]])
                mean = np.nanmean(smoothed_data, axis=0)
                std = np.nanstd(smoothed_data, axis=0)



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

            # plt.plot(time_range, mean)
            plt.plot(time_range, mean, color='black', linewidth=3)
            # plt.fill_between(time_range, mean - std, mean + std, alpha=0.2
            plt.fill_between(time_range, mean - std, mean + std, color='lightgray', alpha=0.2)
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


    """
    Compute each neuron Memory strength:
    correlation score at index 0 -  best correlation score
    """
    neuron_names = np.array(list(row_wise_corr_per_neuron.keys()))
    memory_strength = []
    for neuron_id in row_wise_corr_per_neuron.keys():
        # scores = np.vstack(row_wise_corr_per_neuron[neuron_id])
        scores = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0)[None, :]
        max_scores = np.nanmax(scores, axis=1)
        current_scores = scores[:, 0] # Time steps 0, i.e. the current timesteps
        # strength = np.nanmean(max_scores - current_scores)
        strength = (max_scores - current_scores)
        memory_strength.append(strength)

    print(f'Memory strength std: {np.std(memory_strength, axis=1)}')
    memory_strength = np.mean(memory_strength, axis=1)


    """
    Compute each neuron Memory distance:
    Time index of the peak correlation
    """
    use_wilcoxon = True
    use_tolerance = False
    use_moving_average = True
    tolerance = 0.01
    pvalue_threshold = 0.01
    strength_threshold = 0.01
    distance_threshold = 1
    strong_strength_threshold = 0.05
    far_distance_threshold = 10
    memory_distance = []

    memory_distance = []
    for neuron_id in row_wise_corr_per_neuron.keys():
        mean_scores = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0) # row_wise_corr_per_neuron[neuron_id].shape is (nb_seed * number of model, number_step)
        if use_moving_average:
            # Apply moving average
            smoothed_data = np.array([moving_average(data) for data in row_wise_corr_per_neuron[neuron_id]])
            mean_scores = np.nanmean(smoothed_data, axis=0)

        sorted_indexes = np.array(mean_scores).argsort()[::-1]  # index of highest score to lowest

        if np.isnan(mean_scores).all():
            memory_distance.append(np.nan)
            continue

        # Categorize scores
        distant_peaks = [t for t in sorted_indexes if far_distance_threshold <= t < number_step and not np.isnan(mean_scores[t])]
        near_peaks = [t for t in sorted_indexes if 1 <= t < far_distance_threshold and not np.isnan(mean_scores[t])]
        perceptual_peaks = [0]

        if use_tolerance:
            for peak_list in [distant_peaks, near_peaks]:
                _max_score = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0)[peak_list[0]]
                _distance = peak_list[0]
                for i in range(1, len(peak_list)):
                    score_i = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0)[peak_list[i]]
                    if _distance == 0 or score_i + tolerance < _max_score:
                        # minimum distance or the next biggest score is too small and outside the threshold
                        break
                    if peak_list[i] < _distance and score_i + tolerance >= _max_score:
                        _distance = peak_list[i]
                peak_list.insert(0, _distance)

        distant_scores = np.array(row_wise_corr_per_neuron[neuron_id])[:, distant_peaks[0]]
        near_scores = np.array(row_wise_corr_per_neuron[neuron_id])[:, near_peaks[0]]
        perceptual_scores = np.array(row_wise_corr_per_neuron[neuron_id])[:, perceptual_peaks[0]]

        # Overall peak
        # TODO add tolerance to this version too
        max_distance = sorted_indexes[0]
        max_score = np.array(row_wise_corr_per_neuron[neuron_id])[:, max_distance]

        if use_wilcoxon:
            distant_res = stats.ranksums(distant_scores, perceptual_scores, alternative='greater', nan_policy='omit', axis=None)
            near_res = stats.ranksums(near_scores, perceptual_scores, alternative='greater', nan_policy='omit', axis=None)
            # If score is significantly higher for both near and distant timestep, check which is bigger
            near_over_distant_res = stats.ranksums(near_scores, distant_scores, alternative='greater', nan_policy='omit', axis=None)
            distant_over_near_res = stats.ranksums(distant_scores, near_scores, alternative='greater', nan_policy='omit', axis=None)

            max_res = stats.ranksums(max_score, perceptual_scores, alternative='greater', nan_policy='omit', axis=None)

        if not use_wilcoxon:
            distant_res = stats.ks_2samp(distant_scores, perceptual_scores, alternative='less',
                                         nan_policy='omit', axis=None)
            near_res = stats.ks_2samp(near_scores, perceptual_scores, alternative='less', nan_policy='omit',
                                      axis=None)
            # If score is significantly higher for both near and distant timestep, check which is bigger
            near_over_distant_res = stats.ks_2samp(near_scores, distant_scores, alternative='less',
                                                   nan_policy='omit', axis=None)
            distant_over_near_res = stats.ks_2samp(distant_scores, near_scores, alternative='less',
                                                   nan_policy='omit', axis=None)

            max_res = stats.ks_2samp(max_score, perceptual_scores, alternative='less', nan_policy='omit',
                                     axis=None)

        if False:
            if max_res.pvalue <= pvalue_threshold:
                memory_distance.append(max_distance)
            else:
                memory_distance.append(perceptual_peaks[0])

        if True:
            if distant_res.pvalue <= pvalue_threshold:
                if near_res.pvalue <= pvalue_threshold and near_over_distant_res.pvalue <= pvalue_threshold:
                    memory_distance.append(near_peaks[0])
                else:
                    memory_distance.append(distant_peaks[0])
            elif near_res.pvalue <= pvalue_threshold:
                memory_distance.append(near_peaks[0])
            else:
                memory_distance.append(perceptual_peaks[0])

        if False:
            if near_res.pvalue <= pvalue_threshold:
                if distant_res.pvalue <= pvalue_threshold and distant_over_near_res.pvalue <= pvalue_threshold:
                    memory_distance.append(distant_peaks[0])
                else:
                    memory_distance.append(near_peaks[0])
            elif distant_res.pvalue <= pvalue_threshold:
                memory_distance.append(distant_peaks[0])
            else:
                memory_distance.append(perceptual_peaks[0])

    memory_distance = np.array(memory_distance)


    # Percentage of memory neurons
    long_memory_neurons = memory_distance >= far_distance_threshold
    current_trial_memory_neurons = np.logical_and(memory_distance < far_distance_threshold, memory_distance >= distance_threshold)
    perceptual_neurons = np.abs(memory_distance) < distance_threshold
    memory_neurons = memory_distance >= distance_threshold
    current_trial_future_neurons = np.array([])
    long_future_neurons = np.array([])
    temp_dict = {
        'neuron_ids': neuron_names,
        'memory_strength': memory_strength,
        'memory_distance': memory_distance,
        'perceptual_neurons': perceptual_neurons,
        'current_trial_memory_neurons': current_trial_memory_neurons,
        'long_memory_neurons': long_memory_neurons,
    }

    with open(f"{save_fig_folder}/temporal_selectivity_p{pvalue_threshold}.pickle", 'wb') as handle:
        pickle.dump(temp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('% of perceptual', np.sum(perceptual_neurons)/neuron_names.size)  # perceptual
    print('% of Near memory', np.sum(current_trial_memory_neurons)/neuron_names.size)  # in-trial
    print('% of Distant memory', np.sum(long_memory_neurons)/neuron_names.size) # distant


    print('row-wise done')
