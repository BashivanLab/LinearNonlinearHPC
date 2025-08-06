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

do_element_wise = False
do_row_wise = False
load_previous_row_wise = False
plot_row_wise = False
do_column_wise = False
do_full_matrix = True
load_previous_full_matrix = False


####################################################################################
# Element wise feature importance
# 1. create meta-matrix with 30 steps
#   1.a) Get name of each features
# 2. Select each individuals feature and create a vector with the value of that feature across all samples
# 3. Correlate with firing rates
# 4. Visualize correlation scores in a matrix of 27 x 30
# TODO? Save in txt files: neuron_id, feature_index, feature_name, correlation_score
####################################################################################
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


def plot_matrix(correlation_scores, feature_names, type_analysis, f_name='', max_val=None):
    nx, ny = correlation_scores.shape
    # ny, nx = correlation_scores.shape
    indx, indy = np.arange(nx), np.arange(ny)
    x, y = np.meshgrid(indx, indy)

    fig, ax = plt.subplots()
    fig.set_size_inches(13, 15)
    if max_val is None:
        # im = ax.imshow(correlation_scores, interpolation="nearest", cmap=cm.YlGn)
        im = ax.imshow(correlation_scores.T, interpolation="nearest", cmap=cm.YlGn)
    else:
        im = ax.imshow(correlation_scores.T, interpolation="nearest", cmap=cm.YlGn, vmax=max_val)  # vmin=0,
    # ax.matshow(correlation_scores.T, cmap=cm.YlGn)

    for xval, yval in zip(x.flatten(), y.flatten()):
        score = correlation_scores[xval, yval]
        t = f"{round(score, 2)}"
        if max_val is not None:
            c = 'w' if score > max_val * 0.75 else 'k'  # if dark-green, change text color to white
        else:
            c = 'w' if score > 0.75 else 'k'  # if dark-green, change text color to white
        ax.text(xval, yval, t, color=c, va='center', ha='center', fontsize=11)

    ax.set_xticks(indx + 0.5)
    ax.set_yticks(indy + 0.5)
    ax.grid(ls='-', lw=2)

    for a, ind, labels in zip((ax.xaxis, ax.yaxis), (indx, indy), (feature_names, list(range(n_steps)))):
        a.set_major_formatter(ticker.NullFormatter())
        a.set_minor_locator(ticker.FixedLocator(ind))
        a.set_minor_formatter(ticker.FixedFormatter(labels))

    # ax.xaxis.tick_top()
    plt.title(f"{type_analysis} correlation: {neuron_id}", fontsize=25)
    plt.ylabel('Number of steps in the past', fontsize=20)
    # PCM = ax.get_children()[0]
    # print(ax.get_children())
    cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    # ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=40) # , fontsize=9
    # ax.xaxis.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    # plt.xticks(rotation=90)
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=14)
    plt.setp(ax.yaxis.get_minorticklabels(), fontsize=14)
    # ax.tick_params(axis="x", which="both", rotation=45) # labelsize
    if save_fig:
        os.makedirs(f"{save_fig_folder}/{type_analysis}/", exist_ok=True)
        if max_val is None:
            fig_path = f"{save_fig_folder}/{type_analysis}/{f_name}{neuron_id}.png"
        else:
            fig_path = f"{save_fig_folder}/{type_analysis}/{f_name}{neuron_id}-max{max_val}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400)
        plt.clf()
        plt.close()
    else:
        plt.show()


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

if do_element_wise:
    feature_wise_profile = {}
    nb_f = len(feature_names)
    for neuron_id in metadata.keys():
        correlation_scores = np.zeros((n_steps, nb_f))
        point_biserial_correlation_scores = np.zeros((n_steps, nb_f))
        if aligning_with_activations:
            neuron_metadata, _ = align_metadata_and_activation(metadata, activations, neuron_id)
        else:
            neuron_metadata = metadata[neuron_id]
        X_metadata, firing_rates = get_meta_dataset(neuron_metadata, fr_index, factors_start_index,
                                                    add_trial_phases=add_trial_phases, n_steps=n_steps)
        for n in range(n_steps):
            for f in range(nb_f):
                single_feature_vector = X_metadata[:, f + (n * nb_f)]
                correlation_scores[n, f], _ = pearsonr(single_feature_vector, firing_rates)
                point_biserial_correlation_scores[n, f], _ = pointbiserialr(single_feature_vector, firing_rates)

        plot_matrix(correlation_scores.T, feature_names, 'element-wise')
        plot_matrix(correlation_scores.T, feature_names, 'element-wise_max05', max_val=0.5)
        plot_matrix(correlation_scores.T, feature_names, 'element-wise_max1', max_val=1.0)
        feature_wise_profile[neuron_id] = correlation_scores

    with open(f"{save_fig_folder}/feature_wise_profile.pickle", 'wb') as handle:
        pickle.dump(feature_wise_profile, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def plot_weight_features(feature_reg_weight_profile, neuron_id, feature_names, folder_name, filename, take_abs=True):
    weights = feature_reg_weight_profile[neuron_id].mean(axis=0)
    if take_abs:
        weights = np.abs(feature_reg_weight_profile[neuron_id]).mean(axis=0)
    std = feature_reg_weight_profile[neuron_id].std(axis=0)
    weights, features = zip(*sorted(zip(weights, feature_names)))
    plt.barh(range(len(features)), weights, align='center', xerr=std)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Weight')
    if save_fig:
        os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
        fig_path = f"{save_fig_folder}/{folder_name}/{filename}_{neuron_id}"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400)
        plt.clf()
        plt.close()
    else:
        plt.show()

if do_full_matrix:
    if load_previous_full_matrix:
        with open(f"{save_fig_folder}/feature_reg_weight_profile.pickle", 'rb') as handle:
            feature_reg_weight_profile = pickle.load(handle)
    else:
        regression_files = [
            "reg_output/memory-cc-v2_rnn_3_4-Ideal_obs_k10/344.txt",
            "reg_output/memory-cc-v2_rnn_3_5-Ideal_obs_k10/343.txt",
            "reg_output/memory-cc-v2_rnn_3_6-Ideal_obs_k10/383.txt",
        ]
        nb_steps = [1,1,1]

        feature_reg_weight_profile = {}
        for regression_file in regression_files:
            # for regression_file in regression_files[-1]:
            n = nb_steps[regression_files.index(regression_file)]
            # is_non_linear = 'nonlinear' in regression_file
            filename = folder_path + regression_file
            with open(filename) as file:
                lines = [line.rstrip() for line in file]
            previous_gridsearch_results = lines

            reg_name = 'LinearSVR'
            reg_type = LinearSVR
            reg = reg_type()
            best_parameters = get_regression_params(previous_gridsearch_results, list(metadata.keys())[0], regression_model=reg_name)
            normalize_input = 'reg' in list(best_parameters.keys())[0]
            if not normalize_input:
                reg_model = reg
            else:
                reg_model = Pipeline([('scaler', StandardScaler()), ('reg', reg)])

            for neuron_id in metadata.keys():
                if neuron_id not in feature_reg_weight_profile:
                    feature_reg_weight_profile[neuron_id] = []
                if aligning_with_activations:
                    neuron_metadata, activations = align_metadata_and_activation(metadata, activations, neuron_id)
                else:
                    neuron_metadata = metadata[neuron_id]
                X_metadata, firing_rates = get_meta_dataset(neuron_metadata, fr_index, factors_start_index,
                                                            add_trial_phases=add_trial_phases, n_steps=n)
                best_parameters = get_regression_params(previous_gridsearch_results, neuron_id, regression_model=reg_name)
                reg_model.set_params(**best_parameters)

                predicted_ff, reg_model, kfold_weights = timeseries_crosval_predict(X_metadata, firing_rates, reg_model)
                avg_weights = np.array(kfold_weights).mean(axis=0) # get weights given to each feature across folds
                avg_weights = avg_weights.reshape((n, len(feature_names)))

                folder_and_title_name = 'Regression weights' + f" ({reg_name})"
                folder_and_title_normalized_name = 'Normalized regression weights' + f" ({reg_name})"

                plot_matrix(avg_weights.T, feature_names, folder_and_title_name, f_name=f"{n}-steps-")
                feature_reg_weight_profile[neuron_id].append(avg_weights)

            if n == 30:
                with open(f"{save_fig_folder}/feature_reg_weight_profile.pickle", 'wb') as handle:
                    pickle.dump(feature_reg_weight_profile, handle, protocol=pickle.HIGHEST_PROTOCOL)


        for neuron_id in feature_reg_weight_profile.keys():
            feature_reg_weight_profile[neuron_id] = np.array(feature_reg_weight_profile[neuron_id]).squeeze()
        with open(f"{save_fig_folder}/feature_reg_weight_profile.pickle", 'wb') as handle:
            pickle.dump(feature_reg_weight_profile, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for neuron_id in feature_reg_weight_profile.keys():
        plot_weight_features(feature_reg_weight_profile, neuron_id, feature_names, folder_name='weights_per_neuron', filename='feature_weights', take_abs=False)
        plot_weight_features(feature_reg_weight_profile, neuron_id, feature_names, folder_name='abs_weights_per_neuron', filename='feature_weights', take_abs=True)

    # Average feature importance across neurons
    # Because the scale of the weight varies a lot across neurons (e.g. maximum for a neuron can be 0.015 or 8)
    # Min-max normalize each neuron before averaging across neurons
    abs_feature_weight = {}
    for neuron_id in feature_reg_weight_profile.keys():
        abs_feature_weight[neuron_id] = np.abs(feature_reg_weight_profile[neuron_id])
    norm_abs_feature_weight = {}
    for neuron_id in abs_feature_weight.keys():
        if (np.max(abs_feature_weight[neuron_id]) - np.min(abs_feature_weight[neuron_id])) == 0:
            # For one neuron, svm weights every features with 0
            norm_abs_feature_weight[neuron_id] = abs_feature_weight[neuron_id]
            continue
        norm_abs_feature_weight[neuron_id] = ((abs_feature_weight[neuron_id] - np.min(abs_feature_weight[neuron_id])) /
                                              (np.max(abs_feature_weight[neuron_id]) - np.min(abs_feature_weight[neuron_id])))

    avg_weights_across_neurons = np.array([norm_abs_feature_weight[n] for n in norm_abs_feature_weight.keys()]).mean(axis=1).mean(axis=0)
    std_weights_across_neurons = np.array([norm_abs_feature_weight[n] for n in norm_abs_feature_weight.keys()]).mean(axis=0).std(axis=0)
    folder_name = 'weights_across_neurons'
    filename = 'avg_feature_weights'
    weights, features = zip(*sorted(zip(avg_weights_across_neurons, feature_names)))
    sns.set_theme(context="paper", style='whitegrid', font_scale=1)
    plt.barh(range(len(features)), weights, align='center', xerr=std_weights_across_neurons)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Weight')
    plt.title('Average weight per feature across all neurons')
    if save_fig:
        os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
        fig_path = f"{save_fig_folder}/{folder_name}/{filename}"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400)
        plt.clf()
        plt.close()
    else:
        plt.show()


    ###################################################
    # Number of neurons where a feature is among the top features
    ###################################################
    threshold_important_feature = 0.8
    # Collect the features with a weight above a threshold for each neuron and each seeds
    top_features = [[] for _ in norm_abs_feature_weight]
    for i in range(len(top_features)):
        neuron_id = list(norm_abs_feature_weight.keys())[i]
        seed_i, top_indices = np.where(norm_abs_feature_weight[neuron_id] >= threshold_important_feature)
        neuron_top = [[] for _ in range(norm_abs_feature_weight[neuron_id].shape[0])]
        for j in range(len(seed_i)):
            neuron_top[seed_i[j]].append(top_indices[j])
        top_features[i] = neuron_top
    # Count number of neurons where each feature was in the top (separately per seeds)
    count = np.array([[0 for _ in feature_names]]*3)
    for n in range(len(top_features)):
        for s in range(count.shape[0]):
            count[s][top_features[n][s]] +=1
    percentage_counts = ((np.array(count) / len(norm_abs_feature_weight.keys()))*100)
    avg_perc = np.mean(percentage_counts, axis=0).round()
    std_perc = np.std(percentage_counts, axis=0)
    folder_name = 'weights_across_neurons'
    filename = 'top_features'
    weights, features = zip(*sorted(zip(avg_perc, feature_names)))
    sns.set_theme(context="paper", style='whitegrid', font_scale=1)
    plt.barh(range(len(features)), weights, align='center', xerr=std_perc)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Number of neurons (%)')
    plt.title('Top most important features to predict neural responses')
    if save_fig:
        os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
        fig_path = f"{save_fig_folder}/{folder_name}/{filename}_{threshold_important_feature}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400)
        plt.clf()
        plt.close()
    else:
        plt.show()

    filename = 'top_10_features'
    weights = weights[-10:]
    features = features[-10:]
    sns.set_theme(context="talk", style='whitegrid', font_scale=1)
    plt.barh(range(len(features)), weights, align='center', xerr=std_perc[-10:])
    plt.yticks(range(len(features)), features)
    plt.xlabel('Number of neurons (%)')
    plt.title('Most frequent top features')
    if save_fig:
        os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
        fig_path = f"{save_fig_folder}/{folder_name}/{filename}_{threshold_important_feature}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=400)
        plt.clf()
        plt.close()
    else:
        plt.show()


    ###################################################
    # Plot how many norm_abs_feature_weight >= threshold for each neuron, plot distribution across neurons
    ###################################################
    def plot_histogram(data, xlabel, ylabel, filename, folder_name=None, multiple_names=None, title=None):
        sns.set_theme(context="talk", style='white', font_scale=1)
        sns.despine(top=True)
        if isinstance(data, np.ndarray) and len(data.shape)==2:
            for i in range(data.shape[0]):
                sns.histplot(data[i], stat='percent', bins=[e/10 for e in range(-2, 9, 1)],# bins=[e/10 for e in range(0, 9, 1)], # int(np.ceil(np.nanmax(data)*10)+1)
                             multiple='layer', element='step', alpha=0.4) # , binrange=[0,0.9]
            plt.legend(multiple_names)
        else:
            sns.histplot(data, stat='percent')
        plt.tight_layout()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if title is not None:
            plt.title(title)
        if save_fig:
            if folder_name is not None:
                os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900)
            else:
                plt.savefig(f"{save_fig_folder}/{filename}")
            plt.clf()
            plt.close()
        sns.reset_orig()

    nb_feature_used_per_neurons = []
    threshold_important_feature = 0.8
    # from scipy import stats
    for neuron_id in norm_abs_feature_weight.keys():
        nb_feature_used_per_neurons.append(np.sum(norm_abs_feature_weight[neuron_id] >= threshold_important_feature, axis=1).mean())

    plot_histogram(nb_feature_used_per_neurons, f"Number of important features (weight >= {threshold_important_feature})",
                   "Proportion of neurons (%)", filename=f"nb_important_features_{threshold_important_feature}.png", folder_name=folder_name)

    # Same but use 3*std as threshold instead of abirtraty using a weight
    nb_feature_used_per_neurons = []
    std_important_feature = 2
    for neuron_id in norm_abs_feature_weight.keys():
        if (norm_abs_feature_weight[neuron_id] == 0).all():
            # Skip one neuron where regression fails and all weights are 0
            continue
        above_std_threshold_weight = np.mean(norm_abs_feature_weight[neuron_id]) + (std_important_feature * np.std(norm_abs_feature_weight[neuron_id]))
        nb_feature_used_per_neurons.extend(list(np.sum(norm_abs_feature_weight[neuron_id] >= above_std_threshold_weight, axis=1)))

    filename = f"nb_important_features_{std_important_feature}std.png"
    data = nb_feature_used_per_neurons
    ylabel = "Proportion of neurons (%)"
    xlabel = "Number of important features"
    sns.set_theme(context="talk", style='white', font_scale=1)
    ax = sns.histplot(data, stat='percent', discrete=True)
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    sns.despine(top=True, right=True)
    if save_fig:
        if folder_name is not None:
            os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900)
        else:
            plt.savefig(f"{save_fig_folder}/{filename}")
        plt.clf()
        plt.close()
    print('% neurons with i important feature:', [(i, ((np.array(nb_feature_used_per_neurons)==i).sum() / len(nb_feature_used_per_neurons)).round(2)) for i in np.unique(nb_feature_used_per_neurons)])

    # Same as previous but separate spatially tuned neurons from non-spatially tuned
    dir_path = "rl/figures/spatial_neurons/9nov2024/"
    paths = [dir_path + "memory-cc-v2_rnn_3_6/significant_neurons.pickle",
             dir_path + "memory-cc-v2_rnn_3_5/significant_neurons.pickle",
             dir_path + "memory-cc-v2_rnn_3_4/significant_neurons.pickle",]

    significant_neurons_per_seeds = []
    is_spatially_tuned_dict = {}
    for i, p in enumerate(paths):
        with open(f"{p}", 'rb') as handle:
            seed_significant = pickle.load(handle)
        # Get spatially-tuned neurons (i.e. at least one significant location)
        for n in seed_significant:
            if n not in is_spatially_tuned_dict:
                is_spatially_tuned_dict[n] = []
            is_spatially_tuned_dict[n].append(np.any(list(seed_significant[n].values())))

    nb_feature_used_per_neurons_spatial = []
    nb_feature_used_per_neurons_nonspatial = []
    std_important_feature = 2
    for neuron_id in norm_abs_feature_weight.keys():
        if (norm_abs_feature_weight[neuron_id] == 0).all():
            # Skip one neuron where regression fails and all weights are 0
            continue
        above_std_threshold_weight = np.mean(norm_abs_feature_weight[neuron_id]) + (std_important_feature * np.std(norm_abs_feature_weight[neuron_id]))
        nb_features = list(np.sum(norm_abs_feature_weight[neuron_id] >= above_std_threshold_weight, axis=1))
        for s in range(len(is_spatially_tuned_dict[neuron_id])):
            if is_spatially_tuned_dict[neuron_id][s]:
                nb_feature_used_per_neurons_spatial.append(nb_features[s])
            else:
                nb_feature_used_per_neurons_nonspatial.append(nb_features[s])

    filename = f"nb_important_features_{std_important_feature}std_spatially_tuned.png"
    data_1 = nb_feature_used_per_neurons_spatial
    data_2 = nb_feature_used_per_neurons_nonspatial
    ylabel = "Proportion of neurons (%)"
    xlabel = "Number of important features"
    sns.set_theme(context="talk", style='white', font_scale=1)
    ax = sns.histplot(data_1, stat='percent', discrete=True)
    ax = sns.histplot(data_2, stat='percent', discrete=True)
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    sns.despine(top=True, right=True)
    if save_fig:
        if folder_name is not None:
            os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900)
        else:
            plt.savefig(f"{save_fig_folder}/{filename}")
        plt.clf()
        plt.close()
    print('% spatial neurons with i important feature:', [(i, ((np.array(nb_feature_used_per_neurons_spatial)==i).sum() / len(nb_feature_used_per_neurons_spatial)).round(2)) for i in np.unique(nb_feature_used_per_neurons_spatial)])
    print('% nonspatial neurons with i important feature:', [(i, ((np.array(nb_feature_used_per_neurons_nonspatial)==i).sum() / len(nb_feature_used_per_neurons_nonspatial)).round(2)) for i in np.unique(nb_feature_used_per_neurons_nonspatial)])



    ###################################################
    # Plot accumulation graph
    ###################################################
    sns.set_theme(context="talk", style='white', font_scale=1)
    thresholds = [0.6, 0.7, 0.8, 0.9]
    nb_feature_used_per_neurons = []
    nb_feature_used_per_neurons_std = []
    for neuron_id in norm_abs_feature_weight.keys():
        # nb_feature_per_seed = np.sum(norm_abs_feature_weight[neuron_id] >= t, axis=1)
        nb_feature = [np.sum(norm_abs_feature_weight[neuron_id] >= t, axis=1).mean() for t in thresholds]
        nb_feature_std = [np.sum(norm_abs_feature_weight[neuron_id] >= t, axis=1) for t in thresholds]
        nb_feature_used_per_neurons.append(nb_feature)
        nb_feature_used_per_neurons_std.append(nb_feature_std)
    nb_feature_used_per_neurons = np.array(nb_feature_used_per_neurons)
    nb_feature_used_per_neurons_std = np.array(nb_feature_used_per_neurons_std)
    accumulation = []
    accumulation_std = []
    for x in np.arange(len(feature_names)):
        percentage_with_x_important_features = ((nb_feature_used_per_neurons <= x).sum(axis=0) / len(nb_feature_used_per_neurons) * 100).round()
        accumulation.append(percentage_with_x_important_features)
        accumulation_std.append(((nb_feature_used_per_neurons_std <= x).sum(axis=0) / len(nb_feature_used_per_neurons_std) * 100).std(axis=1))
    accumulation = np.array(accumulation)
    accumulation_std = np.array(accumulation_std)
    first_index_with_100percent = np.sum(~(accumulation == 100).all(axis=1))+1
    accumulation = accumulation[:first_index_with_100percent, :]
    accumulation_std = accumulation_std[:first_index_with_100percent, :]
    x_range = np.arange(accumulation.shape[0])
    fig = plt.plot(x_range, accumulation, label=thresholds, linewidth=1)
    for j in range(accumulation.shape[1]):
        plt.fill_between(x_range, accumulation[:,j] - accumulation_std[:,j], accumulation[:,j] + accumulation_std[:,j], alpha=0.4, color=fig[j].get_color())
    plt.ylabel('Percentage of neurons (%)')
    plt.xlabel('Number of important features')
    plt.xticks(x_range)
    plt.title(f'Number of important features across neurons')
    plt.legend(title='Weight threshold')
    filename='nb_features_accumulation'
    if save_fig:
        if folder_name is not None:
            os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900)
        else:
            plt.savefig(f"{save_fig_folder}/{filename}")
        plt.clf()
        plt.close()
    else:
        plt.show()

    # Same as plot just above but use std to get thresholds
    std_important_feature = 2
    nb_feature_used_per_neurons = []
    nb_feature_used_per_neurons_std = []
    for neuron_id in norm_abs_feature_weight.keys():
        if (norm_abs_feature_weight[neuron_id] == 0).all():
            # Skip one neuron where regression fails and all weights are 0
            continue
        above_std_threshold_weight = np.mean(norm_abs_feature_weight[neuron_id]) + (std_important_feature * np.std(norm_abs_feature_weight[neuron_id]))
        nb_feature_used_per_neurons.append(np.sum(norm_abs_feature_weight[neuron_id] >= above_std_threshold_weight, axis=1))
    nb_feature_used_per_neurons = np.array(nb_feature_used_per_neurons)
    nb_feature_used_per_neurons_std = np.array(nb_feature_used_per_neurons_std)
    accumulation = []
    accumulation_std = []
    for x in np.arange(len(feature_names)):
        percentage_with_x_important_features = ((nb_feature_used_per_neurons <= x).sum(axis=0) / len(nb_feature_used_per_neurons) * 100).round()
        accumulation.append(percentage_with_x_important_features.mean())
        accumulation_std.append(percentage_with_x_important_features.std())

    accumulation = np.array(accumulation)
    accumulation_std = np.array(accumulation_std)
    first_index_with_100percent = np.sum(~(accumulation == 100))+1
    accumulation = accumulation[:first_index_with_100percent]
    accumulation_std = accumulation_std[:first_index_with_100percent]
    x_range = np.arange(accumulation.shape[0])
    fig = plt.plot(x_range, accumulation, label=thresholds, linewidth=1)
    plt.fill_between(x_range, accumulation - accumulation_std, accumulation + accumulation_std) # , alpha=0.4, color=fig[j].get_color()
    plt.ylabel('Cumulative proportion of neurons (%)')
    plt.xlabel('Number of important features')
    plt.xticks(x_range)
    sns.despine(top=True, right=True)
    filename='nb_features_accumulation'
    if save_fig:
        if folder_name is not None:
            os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}_{std_important_feature}std", bbox_inches='tight', dpi=900)
        else:
            plt.savefig(f"{save_fig_folder}/{filename}_{std_important_feature}std")
        plt.clf()
        plt.close()
    else:
        plt.show()


    ###################################################
    # Percentage of spatially-tuned and non-spatial neurons with only one important feature
    #  grouped bar blot where y-axis = proportion of neurons (%); x-axis = weight thresholds; groups = spatially tuned and non-spatial
    ###################################################
    # Load spatial classification of neurons
    # WARNING: number of seeds must match the number of seeds in norm_abs_feature_weight / feature_reg_weight_profile
    dir_path = "rl/figures/spatial_neurons/9nov2024/"
    paths = [dir_path + "memory-cc-v2_rnn_3_6/significant_neurons.pickle",
             dir_path + "memory-cc-v2_rnn_3_5/significant_neurons.pickle",
             dir_path + "memory-cc-v2_rnn_3_4/significant_neurons.pickle",]
    significant_neurons_per_seeds = []
    for i, p in enumerate(paths):
        with open(f"{p}", 'rb') as handle:
            # significant_neurons_seeds.append(pickle.load(handle))
            seed_significant = pickle.load(handle)
        # Get spatially-tuned neurons (i.e. at least one significant location)
        significant_neurons_per_seeds.append(np.array([list(seed_significant[n].values()) for n in seed_significant]).any(axis=1))
    significant_neurons_per_seeds = np.array(significant_neurons_per_seeds)

    # Among neurons in each spatial-group: count the number of neurons with only one important features per threshold
    thresholds = [0.6, 0.7, 0.8, 0.9]
    spatial_group_name = ['Spatially-tuned', 'Non-spatial']
    # neurons_with_one_feature.shape = (nb neurons, nb thresholds, nb seeds)
    neurons_with_one_feature = []
    for neuron_id in norm_abs_feature_weight.keys():
        # only_one_feature.shape == (len(thresholds), nb_seeds=nb seeds in feature_reg_weight_profile/norm_abs_feature_weight)
        only_one_feature = np.array([np.sum(norm_abs_feature_weight[neuron_id] >= t, axis=1) for t in thresholds]) == 1
        neurons_with_one_feature.append(only_one_feature)
    neurons_with_one_feature = np.array(neurons_with_one_feature)
    spatial_neuron_filter = np.repeat(significant_neurons_per_seeds.T[:, None, :], neurons_with_one_feature.shape[1], axis=1)
    nb_one_feat_per_spatial_type = np.empty((len(spatial_group_name), neurons_with_one_feature.shape[2], len(thresholds)))
    nb_one_feat = np.empty((len(spatial_group_name), neurons_with_one_feature.shape[2], len(thresholds)))
    nb_per_spatial_type = np.empty((len(spatial_group_name), neurons_with_one_feature.shape[2], len(thresholds)))
    for t in range(neurons_with_one_feature.shape[1]):
        for s in range(neurons_with_one_feature.shape[2]):
            nb_one_feat_per_spatial_type[0,s,t] = neurons_with_one_feature[:, t, s][spatial_neuron_filter[:, t, s]].sum()
            nb_one_feat_per_spatial_type[1,s,t] = neurons_with_one_feature[:, t, s][(~spatial_neuron_filter)[:, t, s]].sum()
            nb_one_feat[:,s,t] = [neurons_with_one_feature[:, t, s].sum()] * len(spatial_group_name)
            nb_per_spatial_type[0,s,t] = spatial_neuron_filter[:, t, s].sum()
            nb_per_spatial_type[1,s,t] = (~spatial_neuron_filter)[:, t, s].sum()

    grouped_bar_plot(nb_one_feat_per_spatial_type, spatial_group_name, bin_names=thresholds, error_type='std',
                     xlabel='Weight threshold',
                     ylabel='Number of neuron with only one\nimportant feature', filename=f"nb_one_feature",
                     folder_name='spatial_receptive_field', keep_legend=True)

    grouped_bar_plot(nb_one_feat_per_spatial_type/nb_one_feat, spatial_group_name, bin_names=thresholds, error_type='std',
                     xlabel='Weight threshold',
                     ylabel='Distribution of neurons with only one\nimportant feature across spatial selectivity', filename=f"ratio_nb_one_feature",
                     folder_name='spatial_receptive_field', keep_legend=True)

    grouped_bar_plot(nb_one_feat_per_spatial_type/nb_per_spatial_type, spatial_group_name, bin_names=thresholds, error_type='std',
                     xlabel='Weight threshold',
                     ylabel='Proportion of neurons with only one\nimportant feature', filename=f"ratio_nb_one_feature_per_spatial_type",
                     folder_name='spatial_receptive_field', keep_legend=True)

    grouped_bar_plot(nb_one_feat_per_spatial_type/neurons_with_one_feature.shape[0], spatial_group_name, bin_names=thresholds, error_type='std',
                     xlabel='Weight threshold',
                     ylabel='Proportion of neurons with only one\nimportant feature', filename=f"ratio_nb_one_feature_and_spatial_type",
                     folder_name='spatial_receptive_field', keep_legend=True)

    ###################################################
    # Plot neural predictivity histogram for untrained, location, ideal observer
    ###################################################

    filenames = [
        "reg_output/memory-cc-v2_rnn_3_4-Ideal_obs_k10/344.txt",
        "reg_output/memory-cc-v2_rnn_3_5-Ideal_obs_k10/343.txt",
        "reg_output/memory-cc-v2_rnn_3_6-Ideal_obs_k10/383.txt",
        "reg_output/memory-cc-v2_rnn_3_4-Location_k10/344.txt",
        "reg_output/memory-cc-v2_rnn_3_5-Location_k10/343.txt",
        "reg_output/memory-cc-v2_rnn_3_6-Location_k10/383.txt",
    ]
    repeated_seeds = 3
    names = ['Ideal observer', 'Location'] #, 'Untrained'

    reg_name = 'LinearSVR'
    scores = get_regression_model_specific_performance(reg_name, filenames)
    # Concatenate together all seeds of a model
    split_index = [e for e in range(0, len(filenames) + 1, repeated_seeds)]
    data = np.array( [np.array(scores[split_index[i]:split_index[i + 1]]).flatten()
                      for i in range(len(split_index) - 1)])
    assert significant_neurons_per_seeds.shape[0] == repeated_seeds
    spatial_filter = significant_neurons_per_seeds.flatten()
    plot_histogram(data, xlabel="Neural predictivity",
                   ylabel="Proportion of neurons (%)",
                   filename="hist_np", folder_name=None, multiple_names=names,
                   title=None)
    plot_histogram(np.array([data[i][spatial_filter] for i in range(data.shape[0])]), xlabel="Neural predictivity", ylabel="Proportion of neurons (%)",
                   filename="hist_np_spatial_neurons", folder_name=None, multiple_names=names, title="Spatially-tuned neurons")
    plot_histogram(np.array([data[i][~spatial_filter] for i in range(data.shape[0])]), xlabel="Neural predictivity", ylabel="Proportion of neurons (%)",
                   filename="hist_np_nonspatial_neurons", folder_name=None, multiple_names=names, title="Non-spatial neurons")

    ###################################################
    # Plot average weight per class of features
    ###################################################
    grouped_features_1 = ['Location', 'Location', 'Location', 'Location', 'Location', 'Location', 'Location', 'Location', 'Location',
                        'Direction', 'Direction',
                        'Visual information', 'Visual information', 'Visual information', 'Visual information', 'Visual information', 'Visual information', 'Visual information', 'Visual information',
                        'Decision', 'Decision', 'Decision',
                        'Reward', 'Reward', 'Reward',
                        'Correctness', 'Correctness',
                        'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_1 = 'avg_feature_class_weights'

    grouped_features_2 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                        'Spatial', 'Spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial',
                        'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_2 = 'avg_feature_spatial_nonspatial_taskphase_class_weights'

    grouped_features_3 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                        'Spatial', 'Spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial']
    filename_3 = 'avg_feature_spatial_nonspatial_class_weights'
    all_grouped_features = [grouped_features_1, grouped_features_2, grouped_features_3]
    all_filenames = [filename_1, filename_2, filename_3]

    folder_name = 'weights_across_neurons'
    for grouped_features, filename in zip(all_grouped_features, all_filenames):
        group_names = np.unique(grouped_features)
        weights = np.array([norm_abs_feature_weight[n] for n in norm_abs_feature_weight.keys()])
        grouped_feature_weights = []
        grouped_feature_weights_std = []
        for group in group_names:
            ind = [i for i in range(len(grouped_features)) if group == grouped_features[i]]
            avg_weights_per_seeds = weights[:,:,ind].mean(axis=-1).mean(axis=0)
            grouped_feature_weights.append(avg_weights_per_seeds.mean())
            grouped_feature_weights_std.append(avg_weights_per_seeds.std())
        weights, features = zip(*sorted(zip(grouped_feature_weights, group_names)))
        sns.set_theme(context="talk", style='whitegrid', font_scale=1)
        plt.barh(range(len(features)), weights, align='center', xerr=grouped_feature_weights_std)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Weight')
        plt.title('Average weight per type of features across all neurons')
        # plt.title('Maximum weight per class of features across all neurons')
        if save_fig:
            os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
            fig_path = f"{save_fig_folder}/{folder_name}/{filename}"
            plt.savefig(fig_path, bbox_inches='tight', dpi=400)
            plt.clf()
            plt.close()
        else:
            plt.show()

    ###################################################
    # Plot same average weight per class of features as above but separate spatially-tuned and non-spatial neurons
    ###################################################
    sns.set_theme(context="talk", style='white', font_scale=1.2)
    grouped_features_1 = ['Location', 'Location', 'Location', 'Location', 'Location', 'Location', 'Location', 'Location', 'Location',
                        'Direction', 'Direction',
                        'Visual information', 'Visual information', 'Visual information', 'Visual information', 'Visual information', 'Visual information', 'Visual information', 'Visual information',
                        'Decision', 'Decision', 'Decision',
                        'Reward', 'Reward', 'Reward',
                        'Correctness', 'Correctness',
                        'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_1 = 'avg_feature_class_weights_across_spatial_type'
    bin_names_1  = ['Location', 'Direction', 'Visual information', 'Decision', 'Reward', 'Correctness', 'Task phase']

    grouped_features_2 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                        'Spatial', 'Spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial',
                        'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_2 = 'avg_feature_spatial_nonspatial_taskphase_class_weights_across_spatial_type'
    bin_names_2  = ['Spatial features', 'Non-spatial features', 'Task phase']
    bin_names_2  = ['Spatial', 'Non-spatial', 'Task phase']

    grouped_features_3 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                        'Spatial', 'Spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial',
                        'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial']
    filename_3 = 'avg_feature_spatial_nonspatial_class_weights_across_spatial_type'
    bin_names_3 = ['Spatial features', 'Non-spatial features']
    all_grouped_features = [grouped_features_1, grouped_features_2, grouped_features_3]
    all_filenames = [filename_1, filename_2, filename_3]
    all_bin_names = [bin_names_1, bin_names_2, bin_names_3]
    from scipy import stats
    folder_name = 'weights_across_neurons'
    group_names = ['Spatially-tuned neurons', 'Non-spatially tuned']
    for grouped_features, filename, bin_names in zip(all_grouped_features, all_filenames, all_bin_names):
        # Go through grouped features in the same order as the bins (need to have the same names so change it if necessary)
        grouped_features_names = [e.replace(' features', '') for e in bin_names]
        weights = np.array([norm_abs_feature_weight[n] for n in norm_abs_feature_weight.keys()])
        spatial_grouped_feature_weights = []
        spatial_grouped_feature_weights_std = []
        nonspatial_grouped_feature_weights = []
        nonspatial_grouped_feature_weights_std = []
        t_test_spatially_greater = []
        t_test_non_spatially_greater = []
        for group in grouped_features_names:
            t_test_spatially_greater.append([])
            t_test_non_spatially_greater.append([])
            ind = [i for i in range(len(grouped_features)) if group == grouped_features[i]]
            grouped_weights = weights[:,:,ind]
            # spatial_grouped_weights.shape = (number of seeds, number of spatial neurons in this seed, number of features in this group)
            spatial_grouped_weights_per_seeds = []
            nonspatial_grouped_weights_per_seeds = []
            # Loop to keep the seeds separated to compute std across seeds
            for s in range(grouped_weights.shape[1]):
                spatial_grouped_weights_per_seeds.append(grouped_weights[:,s,:][significant_neurons_per_seeds[s]].mean())
                nonspatial_grouped_weights_per_seeds.append(grouped_weights[:,s,:][~significant_neurons_per_seeds[s]].mean())

                res = stats.ranksums(grouped_weights[:,s,:][significant_neurons_per_seeds[s]],
                                grouped_weights[:,(s+1)%grouped_weights.shape[1],:][~significant_neurons_per_seeds[s]],
                                alternative='greater', nan_policy='omit', axis=None)
                t_test_spatially_greater[-1].append(res.pvalue)
                res = stats.ranksums(grouped_weights[:,s,:][~significant_neurons_per_seeds[s]],
                                      grouped_weights[:, (s+1)%grouped_weights.shape[1], :][significant_neurons_per_seeds[s]],
                                    alternative='greater', nan_policy='omit', axis=None)
                t_test_non_spatially_greater[-1].append(res.pvalue)

            # Across all seeds
            spatially_tuned_filter = np.repeat(significant_neurons_per_seeds.T[:,:,None], grouped_weights.shape[2],axis=2)
            res = stats.ranksums(grouped_weights[spatially_tuned_filter],
                                 grouped_weights[~spatially_tuned_filter],
                                 alternative='greater', nan_policy='omit', axis=None)
            print(f"{group} spatially greater: {res.pvalue}")
            res = stats.ranksums(grouped_weights[~spatially_tuned_filter],
                                 grouped_weights[spatially_tuned_filter],
                                 alternative='greater', nan_policy='omit', axis=None)
            print(f"{group} Non-spatially greater: {res.pvalue}")

            spatial_grouped_feature_weights.append(np.mean(spatial_grouped_weights_per_seeds))
            spatial_grouped_feature_weights_std.append(np.std(spatial_grouped_weights_per_seeds))
            nonspatial_grouped_feature_weights.append(np.mean(nonspatial_grouped_weights_per_seeds))
            nonspatial_grouped_feature_weights_std.append(np.std(nonspatial_grouped_weights_per_seeds))

        data=np.vstack((spatial_grouped_feature_weights, nonspatial_grouped_feature_weights))
        data_std=np.vstack((spatial_grouped_feature_weights_std, nonspatial_grouped_feature_weights_std))
        df = {
            'data': data.flatten(),
            'neuron_type': np.tile(group_names, (len(bin_names), 1)).T.flatten(),
            'bin_name': np.tile(bin_names, (len(group_names), 1)).flatten(),
            'std': data_std.flatten(),
        }
        df = pd.DataFrame(df)
        m = 'data'
        error_type='std'
        ax = sns.barplot(data=df, x='bin_name', y=m, hue='neuron_type',
                         palette=sns.set_palette(sns.color_palette("deep")))
        plt.xlabel('Features')
        plt.ylabel('Average weight')
        ax.legend().set_title('')
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        plt.errorbar(x=x_coords, y=y_coords, yerr=df[error_type], fmt="none", c="k", capsize=3)  # , elinewidth=0.5
        plt.tight_layout()
        sns.despine(top=True, right=True)
        if save_fig:
            if folder_name is not None:
                os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900)
            else:
                plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', dpi=900)
                # plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', format='svg')
            plt.clf()
            plt.close()
        else:
            plt.show()

    # Same but boxplot
    grouped_features_1 = ['Location', 'Location', 'Location', 'Location', 'Location', 'Location', 'Location', 'Location',
                          'Location',
                          'Direction', 'Direction',
                          'Visual information', 'Visual information', 'Visual information', 'Visual information',
                          'Visual information', 'Visual information', 'Visual information', 'Visual information',
                          'Decision', 'Decision', 'Decision',
                          'Reward', 'Reward', 'Reward',
                          'Correctness', 'Correctness',
                          'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_1 = 'avg_feature_class_weights_across_spatial_type'
    bin_names_1 = ['Location', 'Direction', 'Visual information', 'Decision', 'Reward', 'Correctness', 'Task phase']

    grouped_features_2 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                          'Spatial', 'Spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial',
                          'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_2 = 'avg_feature_spatial_nonspatial_taskphase_class_weights_across_spatial_type'
    bin_names_2 = ['Spatial', 'Non-spatial', 'Task phase']

    grouped_features_3 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                          'Spatial', 'Spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial']
    filename_3 = 'avg_feature_spatial_nonspatial_class_weights_across_spatial_type'
    bin_names_3 = ['Spatial features', 'Non-spatial features']

    grouped_features_4 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                          'Spatial',
                          'Spatial', 'Spatial',
                          'Visual', 'Visual', 'Visual', 'Visual',
                          'Visual', 'Visual', 'Visual', 'Visual',
                          'Decision', 'Decision', 'Decision',
                          'Reward', 'Reward', 'Reward',
                          'Reward', 'Reward',
                          'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_4 = 'avg_feature_spatial_nonspatial_iogroups_class_weights_across_spatial_type'
    bin_names_4 = ['Spatial', 'Task phase', 'Visual', 'Reward', 'Decision']

    # all_grouped_features = [grouped_features_1, grouped_features_2, grouped_features_3, grouped_features_4]
    # all_filenames = [filename_1, filename_2, filename_3, filename_4]
    # all_bin_names = [bin_names_1, bin_names_2, bin_names_3, bin_names_4]
    all_grouped_features = [grouped_features_4]
    all_filenames = [filename_4]
    all_bin_names = [bin_names_4]

    from scipy import stats

    folder_name = 'weights_across_neurons'
    group_names = ['Spatially tuned', 'Non-spatially tuned']

    for grouped_features, filename, bin_names in zip(all_grouped_features, all_filenames, all_bin_names):
        # Go through grouped features in the same order as the bins (need to have the same names so change it if necessary)
        grouped_features_names = [e.replace(' features', '') for e in bin_names]
        weights = np.array([norm_abs_feature_weight[n] for n in norm_abs_feature_weight.keys()])

        # Prepare lists to store all individual data points
        all_spatial_weights = []
        all_nonspatial_weights = []
        p_values_spatial_greater = []
        p_values_nonspatial_greater = []

        for group in grouped_features_names:
            ind = [i for i in range(len(grouped_features)) if group == grouped_features[i]]
            grouped_weights = weights[:, :, ind]

            # Flatten all data points for this group
            spatial_weights = []
            nonspatial_weights = []

            for s in range(grouped_weights.shape[1]):
                spatial_weights.extend(grouped_weights[:, s, :][significant_neurons_per_seeds[s]].flatten())
                nonspatial_weights.extend(grouped_weights[:, s, :][~significant_neurons_per_seeds[s]].flatten())

            all_spatial_weights.append(spatial_weights)
            all_nonspatial_weights.append(nonspatial_weights)

            # Perform statistical test
            # Using Mann-Whitney U test (non-parametric alternative to t-test)
            try:
                stat, p = stats.mannwhitneyu(spatial_weights, nonspatial_weights, alternative='greater')
                p_values_spatial_greater.append(p)
            except:
                p_values_spatial_greater.append(np.nan)

            try:
                stat, p = stats.mannwhitneyu(nonspatial_weights, spatial_weights, alternative='greater')
                p_values_nonspatial_greater.append(p)
            except:
                p_values_nonspatial_greater.append(np.nan)

        # Create a DataFrame for plotting
        plot_data = []
        for i, group in enumerate(bin_names):
            plot_data.extend(
                [{'weight': w, 'group': group, 'neuron_type': 'Spatially-tuned'} for w in all_spatial_weights[i]])
            plot_data.extend(
                [{'weight': w, 'group': group, 'neuron_type': 'Non-spatially tuned'} for w in all_nonspatial_weights[i]])

        df = pd.DataFrame(plot_data)

        # Create the boxplot with individual points
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=df, x='group', y='weight', hue='neuron_type',
                         palette=sns.set_palette(sns.color_palette("deep")),
                         showfliers=False)  # Hide outliers since we'll show all points

        plt.xlabel('Features')
        plt.ylabel('Normalized weight')
        ax.legend().set_title('')

        # Add significance markers
        y_max = df['weight'].max()
        y_step = y_max * 0.05
        current_y = y_max + y_step

        for i, group in enumerate(bin_names):
            # Spatial vs non-spatial comparison
            p_spatial = p_values_spatial_greater[i]
            p_nonspatial = p_values_nonspatial_greater[i]

            if p_spatial < 0.05:
                plt.text(i, current_y, '*', ha='center', va='center', color='black', fontsize=12)
                plt.hlines(y=current_y - y_step * 0.3, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)
                current_y += y_step
            if p_nonspatial < 0.05:
                plt.text(i, current_y, '*', ha='center', va='center', color='black', fontsize=12)
                plt.hlines(y=current_y - y_step * 0.3, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)
                current_y += y_step

        plt.ylim(top=current_y + y_step)
        plt.tight_layout()
        sns.despine(top=True, right=True)

        if save_fig:
            if folder_name is not None:
                os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900)
            else:
                plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', dpi=900)
            plt.clf()
            plt.close()
        else:
            plt.show()


#FIGURE Mirrored HISTOGRAM
    grouped_features_2 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                          'Spatial', 'Spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial', 'Non-spatial',
                          'Non-spatial', 'Non-spatial',
                          'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_2 = 'avg_feature_spatial_nonspatial_taskphase_class_weights_across_spatial_type'
    bin_names_2 = ['Spatial', 'Non-spatial', 'Task phase']

    grouped_features_4 = ['Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial', 'Spatial',
                          'Spatial',
                          'Spatial', 'Spatial',
                          'Visual', 'Visual', 'Visual', 'Visual',
                          'Visual', 'Visual', 'Visual', 'Visual',
                          'Decision', 'Decision', 'Decision',
                          'Reward', 'Reward', 'Reward',
                          'Reward', 'Reward',
                          'Task phase', 'Task phase', 'Task phase', 'Task phase', 'Task phase']
    filename_4 = 'avg_feature_spatial_nonspatial_iogroups_class_weights_across_spatial_type'
    bin_names_4 = ['Spatial', 'Task phase', 'Visual', 'Reward', 'Decision']

    all_grouped_features = [grouped_features_2, grouped_features_4]
    all_filenames = [filename_2, filename_4]
    all_bin_names = [bin_names_2, bin_names_4]

    from scipy import stats
    from matplotlib.patches import Patch

    folder_name = 'weights_across_neurons'
    group_names = ['Spatially tuned neurons', 'Non-spatially tuned']

    for grouped_features, filename, bin_names in zip(all_grouped_features, all_filenames, all_bin_names):
        # Go through grouped features in the same order as the bins (need to have the same names so change it if necessary)
        grouped_features_names = [e.replace(' features', '') for e in bin_names]
        weights = np.array([norm_abs_feature_weight[n] for n in norm_abs_feature_weight.keys()])

        # Prepare lists to store all individual data points
        all_spatial_weights = []
        all_nonspatial_weights = []
        p_values_spatial_greater = []
        p_values_nonspatial_greater = []

        for group in grouped_features_names:
            ind = [i for i in range(len(grouped_features)) if group == grouped_features[i]]
            grouped_weights = weights[:, :, ind]

            # Flatten all data points for this group
            spatial_weights = []
            nonspatial_weights = []

            for s in range(grouped_weights.shape[1]):
                spatial_weights.extend(grouped_weights[:, s, :][significant_neurons_per_seeds[s]].flatten())
                nonspatial_weights.extend(grouped_weights[:, s, :][~significant_neurons_per_seeds[s]].flatten())

            all_spatial_weights.append(spatial_weights)
            all_nonspatial_weights.append(nonspatial_weights)

            try:
                stat, p = stats.ranksums(spatial_weights, nonspatial_weights, alternative='greater')
                p_values_spatial_greater.append(p)
            except:
                p_values_spatial_greater.append(np.nan)

            try:
                stat, p = stats.ranksums(nonspatial_weights, spatial_weights, alternative='greater')
                p_values_nonspatial_greater.append(p)
            except:
                p_values_nonspatial_greater.append(np.nan)

        # Create a DataFrame for plotting
        plot_data = []
        for i, group in enumerate(bin_names):
            plot_data.extend(
                [{'weight': w, 'group': group, 'neuron_type': 'Spatially-tuned'} for w in all_spatial_weights[i]])
            plot_data.extend(
                [{'weight': w, 'group': group, 'neuron_type': 'Non-spatially tuned'} for w in all_nonspatial_weights[i]])

        df = pd.DataFrame(plot_data)

        # Create the mirrored histogram plot
        sns.set_theme(context="talk", style='white', font_scale=1.2)
        plt.figure(figsize=(10, 6))

        def mirrored_hist(data, color=None, label=None, **kwargs):
            spatial_data = data[data['neuron_type'] == 'Spatially-tuned']['weight']
            nonspatial_data = data[data['neuron_type'] == 'Non-spatially tuned']['weight']

            # Calculate bins based on the combined data range
            combined = np.concatenate([spatial_data, nonspatial_data])
            bins = np.histogram_bin_edges(combined, bins=30)

            # Calculate means
            spatial_mean = np.mean(spatial_data)
            nonspatial_mean = np.mean(nonspatial_data)

            # Plot spatially-tuned facing LEFT (using negative counts)
            counts, bins = np.histogram(spatial_data, bins=bins, density=True)
            plt.barh(bins[:-1], -counts, height=np.diff(bins), align='edge',
                     color=sns.color_palette("deep")[0],  # alpha=0.7,
                     edgecolor=None, linewidth=0.5)

            # Add mean line for spatial (facing left)
            plt.axhline(y=spatial_mean, color='gray',
                        # linestyle='--', linewidth=1.5, xmin=0.05, xmax=0.45)
                        linestyle='--', linewidth=1.5, xmin=0.15, xmax=0.35)

            # Plot non-spatially tuned facing RIGHT
            counts, bins = np.histogram(nonspatial_data, bins=bins, density=True)
            plt.barh(bins[:-1], counts, height=np.diff(bins), align='edge',
                     color=sns.color_palette("deep")[1],  # alpha=0.7,
                     edgecolor=None, linewidth=0.5)

            # Add mean line for non-spatial (facing right)
            plt.axhline(y=nonspatial_mean, color='gray',
                        # linestyle='--', linewidth=1.5, xmin=0.55, xmax=0.95)
                        linestyle='--', linewidth=1.5, xmin=0.65, xmax=0.85)

        # Create the grid
        g = sns.FacetGrid(df, col='group', col_order=bin_names, height=6, aspect=0.5,
                          sharex=False, sharey=True)

        # Map our custom histogram function
        g.map_dataframe(mirrored_hist)

        # Adjust axes to show mirrored effect
        for ax in g.axes.flat:
            # Set x-axis to cross at zero
            ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)

            # Adjust x-axis limits symmetrically
            max_density = max([abs(x) for x in ax.get_xlim()])
            ax.set_xlim(-max_density, max_density)

            # Add group title
            ax.set_title(ax.get_title().replace('group = ', ''))

            # Add proper labels
            if ax.is_first_col():
                ax.set_ylabel('Normalized weight')
            ax.set_xlabel('Density')

            # Add x-axis ticks on both sides
            ax.xaxis.set_ticks_position('both')

        # Add legend
        legend_elements = [Patch(facecolor=sns.color_palette("deep")[0], label='Spatially-tuned'),
                           Patch(facecolor=sns.color_palette("deep")[1], label='Non-spatially tuned')]
        g.fig.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        if save_fig:
            if folder_name is not None:
                os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}.svg", bbox_inches='tight', format='svg')
            else:
                plt.savefig(f"{save_fig_folder}/{filename}.svg", bbox_inches='tight', format='svg')
            plt.clf()
            plt.close()
        else:
            plt.show()

        # Print p-values for each feature group
        print("Statistical comparison of feature weights between neuron types:")
        print("(Wilcoxon rank sum test, one-sided)")
        print("--------------------------------------------------")
        print("{:<15} {:<30} {:<30}".format("Feature", "p-value (Spatial > Non-spatial)", "p-value (Non-spatial > Spatial)"))
        print("--------------------------------------------------")

        for i, group in enumerate(bin_names):
            spatial_p = p_values_spatial_greater[i]
            nonspatial_p = p_values_nonspatial_greater[i]

            # Format p-values with significance stars
            def format_p(p):
                if np.isnan(p):
                    return "N/A"
                if p < 0.001:
                    return f"{p} ***"
                elif p < 0.01:
                    return f"{p} **"
                elif p < 0.05:
                    return f"{p} *"
                else:
                    return f"{p}"


            print("{:<15} {:<30} {:<30}".format(
                group,
                format_p(spatial_p),
                format_p(nonspatial_p)
            ))

        print("--------------------------------------------------")
        print("*** p < 0.001, ** p < 0.01, * p < 0.05")


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
            row_wise_corr_per_neuron = pickle.load(handle)

    else:
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
                    # row_wise_corr_per_neuron[neuron_id] = correlation_per_row
                    row_wise_corr_per_neuron[neuron_id].append(np.array(correlation_per_row))

        with open(f"{save_fig_folder}/row_wise_profile.pickle", 'wb') as handle:
            pickle.dump(row_wise_corr_per_neuron, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('.')

    if plot_row_wise:
        for neuron_id in row_wise_corr_per_neuron.keys():
            if isinstance(number_step, tuple):
                time_range = [i for i in range(number_step[1], -number_step[0], -1)]
            else:
                time_range = [i for i in range(number_step)]
            sns.set_theme(context="talk", style='white', font_scale=1.2)
            mean = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0)
            std = np.nanstd(row_wise_corr_per_neuron[neuron_id], axis=0)
            plt.plot(time_range, mean)
            plt.fill_between(time_range, mean - std, mean + std, alpha=0.2)
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
        scores = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0)[None, :]
        max_scores = np.nanmax(scores, axis=1)
        current_scores = scores[:, 0] # Time steps 0, i.e. the current timesteps
        strength = (max_scores - current_scores)
        memory_strength.append(strength)

    print(f'Memory strength std: {np.std(memory_strength, axis=1)}')
    memory_strength = np.mean(memory_strength, axis=1)

    """
    Compute each neuron Memory distance:
    Time index of the peak correlation
    If the difference between the highest peak and a peak closer to step 0 is below a tolerance (e.g. 0.01),
        Select the time index of the peak closest to present time
    """
    tolerance = 0.01
    memory_distance = []
    for neuron_id in row_wise_corr_per_neuron.keys():
        scores = np.nanmean(row_wise_corr_per_neuron[neuron_id], axis=0)
        sorted_indexes = np.array(scores).argsort()[::-1]  # index of highest score to lowest
        max_score = scores[sorted_indexes[0]]
        distance = sorted_indexes[0]
        for i in range(1, len(sorted_indexes)):
            if distance == 0 or scores[sorted_indexes[i]] + tolerance < max_score:
                # minimum distance or the next biggest score is too small and outside the threshold
                break
            if sorted_indexes[i] < distance and scores[sorted_indexes[i]] + tolerance >= max_score:
                distance = sorted_indexes[i]
        if np.isnan(max_score):
            distance = np.nan
        memory_distance.append(distance)
    memory_distance = np.array(memory_distance)

    # Percentage of memory neurons
    strength_threshold = 0.01
    distance_threshold = 1
    strong_strength_threshold = 0.05
    far_distance_threshold = 10

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

    with open(f"{save_fig_folder}/temporal_selectivity_tol{tolerance}.pickle", 'wb') as handle:
        pickle.dump(temp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('% of perceptual', np.sum(perceptual_neurons)/neuron_names.size)
    print('% of Near memory', np.sum(current_trial_memory_neurons)/neuron_names.size)
    print('% of Distant memory', np.sum(long_memory_neurons)/neuron_names.size)


    """
    Load models score
    """
    metric = 'Corr'
    filenames = [
        "reg_output/memory-cc-v2_rnn_3_4-k10/0.txt",
        "reg_output/memory-cc-v2_rnn_3_5-k10/0.txt",
        "reg_output/memory-cc-v2_rnn_3_6-k10/0.txt",
        "reg_output/memory-cc-v2_rnn_3_4-Location_k10/344.txt",
        "reg_output/memory-cc-v2_rnn_3_5-Location_k10/343.txt",
        "reg_output/memory-cc-v2_rnn_3_6-Location_k10/383.txt",
        "reg_output/memory-cc-v2_rnn_3_4-Ideal_obs_k10/344.txt",
        "reg_output/memory-cc-v2_rnn_3_5-Ideal_obs_k10/343.txt",
        "reg_output/memory-cc-v2_rnn_3_6-Ideal_obs_k10/383.txt",
        "reg_output/memory-cc-v2_epnam_14_cl_1-k10.txt",
        "reg_output/memory-cc-v2_epnam_14_cl_2-k10.txt",
        "reg_output/memory-cc-v2_epnam_14_cl_3-k10.txt",
        "reg_output/memory-cc-v2_rnn_3_4-k10/344.txt",
        "reg_output/memory-cc-v2_rnn_3_5-k10/343.txt",
        "reg_output/memory-cc-v2_rnn_3_6-k10/383.txt",
    ]
    repeated_seeds = 3
    names = ['Untrained ROARN', 'Location', 'Ideal observer', 'EPN', 'ROARN']
    reg_name = 'LinearSVR'
    scores = get_regression_model_specific_performance(reg_name, filenames)

    if isinstance(repeated_seeds, list):
        splitting_index = [0]
        for i in range(len(repeated_seeds)):
            splitting_index.append(splitting_index[-1] + repeated_seeds[i])
        if splitting_index[-1] == len(scores):
            # Remove the last splitting index because it would create an empty array after split
            splitting_index = splitting_index[1:-1]
        else:
            splitting_index = splitting_index[1:]
        scores_grouped_seeds = np.array_split(scores, splitting_index)
    elif repeated_seeds > 1:
        scores_grouped_seeds = np.array_split(scores, [i for i in range(repeated_seeds, len(scores), repeated_seeds)])

    """
    Scatter plot
    x = Memory Strength
    y = Memory distance
    color = Correlation score by best model
    """
    sns.set_theme(context="talk", style="white", font_scale=1.2)
    names.append("Mnemonic properties")
    for model_type in names:
        without_nan = np.logical_or(np.isnan(memory_strength), np.isnan(memory_distance)) == False
        if 'Mnemonic properties' in model_type:
            corr, pvalue = pearsonr(memory_strength[without_nan], memory_distance[without_nan])
            # plt.annotate("$\it{r}$"+f" = {corr:.2f}", (0.01, 0.94), xycoords='axes fraction')
            plt.annotate("$\it{r}$"+f" = {corr:.2f}", (0.01, 0.91), xycoords='axes fraction')
            model_scores = np.ones(without_nan.shape)
        else:
            model_scores = scores_grouped_seeds[names.index(model_type)]
            model_scores = np.clip(np.nanmean(model_scores, axis=0), 0 ,1) # To use them as color, set between 0 and 1.0
        model_scores = model_scores[without_nan]
        zorder = np.argsort(
            model_scores)  # plot scatter points in reverse order of colors so that white dots doesn't hide overlapping green ones
        plt.scatter(memory_distance[zorder], memory_strength[zorder], c=model_scores[zorder], cmap="BuGn")
        plt.clim(0.0, 1.0)
        cb = plt.colorbar(label="Neural Predictivity")
        cb.outline.set_visible(False)
        plt.xlabel('Memory Distance')
        plt.ylabel('Memory Strength')
        plt.title(model_type)
        sns.despine(top=True, right=True)
        if save_fig:
            os.makedirs(f"{save_fig_folder}", exist_ok=True)
            os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)
            fig_path = f"{save_fig_folder}/memory-scatter-{reg_name}-{model_type}-distance-tol-{tolerance}.png"
            plt.savefig(fig_path, bbox_inches='tight', dpi=900)
            plt.savefig(f"{save_fig_folder}/svg/memory-scatter-{reg_name}-{model_type}-distance-tol-{tolerance}.svg", bbox_inches='tight', format='svg')
            plt.clf()
            plt.close()
        else:
            plt.show()
    names = names[:-1]


    # Mnemonic properties
    without_nan = np.logical_or(np.isnan(memory_strength), np.isnan(memory_distance)) == False
    corr, pvalue = pearsonr(memory_strength[without_nan], memory_distance[without_nan])
    plt.annotate(r"$\rho$" + f" = {corr:.2f}", xy=(0.76, 0.91), xycoords='axes fraction')
    plt.scatter(memory_distance, memory_strength)
    cb.outline.set_visible(False)
    plt.xlabel('Memory Distance')
    plt.ylabel('Memory Strength')
    plt.title(model_type)
    sns.despine(top=True, right=True)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.round(np.linspace(max(ymin,0), ymax, 3), 2))
    if save_fig:
        os.makedirs(f"{save_fig_folder}", exist_ok=True)
        os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)
        fig_path = f"{save_fig_folder}/memory-scatter-{reg_name}-{model_type}-distance-tol-{tolerance}.png"
        plt.savefig(fig_path, bbox_inches='tight', dpi=900)
        plt.savefig(f"{save_fig_folder}/svg/memory-scatter-{reg_name}-{model_type}-distance-tol-{tolerance}.svg",
                    bbox_inches='tight', format='svg')
        plt.clf()
        plt.close()
    else:
        plt.show()

    """
    Scatter plot correlation scores of multiple models vs memory distance
    """
    def export_legend(ax, save_fig_folder):
        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        ax2.axis('off')
        legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc="lower left", ncol=4)
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(f"{save_fig_folder}/legend.png", bbox_inches=bbox, dpi=1000)
        plt.savefig(f"{save_fig_folder}/legend.svg", bbox_inches=bbox, dpi=1000, format='svg')
    """
    Grouped bar plot correlation scores vs memory properties of neuron
    Shows how well mnemonic and non-mnemonic neurons are predicted
    """
    from scipy.stats import binned_statistic, sem, rankdata

    save_legend = False
    percentile_bin = False

    sns.set_theme(context="talk", style='whitegrid', font_scale=1.2)
    sns.set_style("whitegrid")
    for mem_type in ['Memory Distance', 'Memory Strength']:
        if mem_type == 'Memory Distance':
            nb_bin = 5
            binrange = (0, 30)
            mem_values = memory_distance# percentile
        else:
            nb_bin = 5
            binrange = (0.0, 0.1)
            mem_values = memory_strength

        if percentile_bin:
            nb_bin = 4
            binrange = (0.0, 1.0)
            mem_values = rankdata(mem_values, "average") / len(mem_values)

        subselection_names_plot = np.array(['Location', 'Ideal observer', 'Ideal observer with memory', 'EPN', 'ROARN'])
        all_bin_mean_seeds = [[] for _ in range(int(len(subselection_names)/repeated_seeds))]
        all_bin_median_seeds = [[] for _ in range(int(len(subselection_names)/repeated_seeds))]

        all_model_types = [[] for _ in range(int(len(subselection_names)/repeated_seeds))]
        all_bin_id = [[] for _ in range(int(len(subselection_names)/repeated_seeds))]

        for model_type in subselection_names:
            model_scores = np.array(scores)[names.index(model_type)]
            model_scores = model_scores[without_nan]
            without_nan = np.logical_or(np.isnan(mem_values), np.isnan(model_scores)) == False
            bin_mean = binned_statistic(mem_values[without_nan], model_scores[without_nan], statistic=np.nanmean,
                                        bins=nb_bin, range=binrange)
            bin_median = binned_statistic(mem_values[without_nan], model_scores[without_nan], statistic=np.nanmedian,
                                          bins=nb_bin, range=binrange)

            model_index = int(subselection_names.index(model_type) / repeated_seeds)
            all_bin_mean_seeds[model_index].append(bin_mean.statistic)
            all_bin_median_seeds[model_index].append(bin_median.statistic)

            all_model_types[model_index] = [subselection_names_plot[model_index] for i in range(nb_bin)]
            if mem_type == 'Memory Distance':
                if percentile_bin:
                    all_bin_id[model_index] = [f"{round(bin_mean.bin_edges[i], 2)}-{round(bin_mean.bin_edges[i + 1], 2)}" for i in range(nb_bin)]
                else:
                    all_bin_id[model_index] = [f"{int(bin_mean.bin_edges[i])}-{int(bin_mean.bin_edges[i + 1])}" for i in range(nb_bin)]
            else:
                all_bin_id[model_index] = [f"{round(bin_mean.bin_edges[i], 2)}-{round(bin_mean.bin_edges[i + 1], 2)}" for i in range(nb_bin)]

        all_bin_mean = np.round(np.nanmean(all_bin_mean_seeds, axis=1), 2)
        all_bin_mean_std = np.round(np.nanstd(all_bin_mean_seeds, axis=1), 4)
        all_bin_median = np.round(np.nanmean(all_bin_median_seeds, axis=1), 2)
        all_bin_median_std = np.round(np.nanstd(all_bin_median_seeds, axis=1), 4)
        df = {
            'mean': all_bin_mean.flatten(),
            'median': all_bin_median.flatten(),
            'mean_std': all_bin_mean_std.flatten(),
            'median_std': all_bin_median_std.flatten(),
            'model_type': np.array(all_model_types).flatten(),
            'bin_name': np.array(all_bin_id).flatten()
        }
        df = pd.DataFrame(df)
        for m in ['mean', 'median']:
            ax = sns.barplot(data=df, x='bin_name', y=m, hue='model_type', palette=sns.set_palette(sns.color_palette("deep")))
            plt.ylabel(f'{str(m).title()} Neural Predictivity')
            plt.xlabel(mem_type)
            plt.ylim(0.0, round(np.nanmax(df[m]), 2) + 0.02)
            sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, -0.35), ncol=4, title=None, frameon=False)
            # if save_legend:
            #     export_legend(ax, f"{save_fig_folder}/{experiment_name}")
            ax.get_legend().remove()
            x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
            y_coords = [p.get_height() for p in ax.patches]
            if m == 'mean':
                er = 'mean_std'
            else:
                er = 'median_std'
            plt.errorbar(x=x_coords, y=y_coords, yerr=df[er], fmt="none", c="k", capsize=3) #, elinewidth=0.5
            plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
            plt.tight_layout()
            if save_fig:
                os.makedirs(f"{save_fig_folder}/", exist_ok=True)
                if mem_type == 'Memory Distance':
                    fig_path = f"{save_fig_folder}/memory-barplot-{reg_name}-distance-tol-{tolerance}-{m}-distance.png"
                else:
                    fig_path = f"{save_fig_folder}/memory-barplot-{reg_name}-{m}-strength.png"
                plt.savefig(fig_path, bbox_inches='tight', dpi=900)
                plt.savefig(fig_path[:-3]+'svg', bbox_inches='tight', format='svg')
                plt.clf()
                plt.close()
            else:
                plt.show()


    categories = [("Distant Memory", long_memory_neurons), ("Near Memory", current_trial_memory_neurons), ("Perceptual", perceptual_neurons)]
    subselection_names_plot = np.array(['Untrained ROARN', 'Location', 'Ideal observer', 'EPN', 'ROARN'])

    all_bin_mean_seeds = [[] for _ in range(int(len(subselection_names) / repeated_seeds))]
    all_bin_median_seeds = [[] for _ in range(int(len(subselection_names) / repeated_seeds))]
    all_model_types = [[] for _ in range(int(len(subselection_names) / repeated_seeds))]
    all_bin_id = [[] for _ in range(int(len(subselection_names) / repeated_seeds))]
    for model_type in subselection_names:
        model_scores = np.array(scores)[names.index(model_type)]
        model_scores = model_scores[without_nan]
        model_index = int(subselection_names.index(model_type) / repeated_seeds)

        if len(categories) == 3:
            means = [np.nanmean(model_scores[perceptual_neurons]), np.nanmean(model_scores[current_trial_memory_neurons]), np.nanmean(model_scores[long_memory_neurons])]
            medians = [np.nanmedian(model_scores[perceptual_neurons]), np.nanmedian(model_scores[current_trial_memory_neurons]), np.nanmedian(model_scores[long_memory_neurons])]

        if len(categories) == 2:
            means = [np.nanmean(model_scores[perceptual_neurons]), np.nanmean(model_scores[memory_neurons])]
            medians = [np.nanmedian(model_scores[perceptual_neurons]), np.nanmedian(model_scores[memory_neurons])]

        all_bin_id[model_index] = categories
        all_model_types[model_index] = [subselection_names_plot[model_index] for i in range(len(categories))]

        if len(categories) not in [2, 3]:
            means = [np.nanmean(e[1]) for e in categories]
            medians = [np.nanmedian(e[1]) for e in categories]
            all_bin_id[model_index] = [e[0] for e in categories]

        all_bin_mean_seeds[model_index].append(means)
        all_bin_median_seeds[model_index].append(medians)

    all_bin_mean = np.round(np.nanmean(all_bin_mean_seeds, axis=1), 2)
    all_bin_mean_std = np.round(np.nanstd(all_bin_mean_seeds, axis=1), 4)
    all_bin_median = np.round(np.nanmean(all_bin_median_seeds, axis=1), 2)
    all_bin_median_std = np.round(np.nanstd(all_bin_median_seeds, axis=1), 4)

    rnn_normed_bin_mean = np.copy(all_bin_mean)
    rnn_normed_bin_median = np.copy(all_bin_median)
    untrained_normed_bin_mean = np.copy(all_bin_mean)
    untrained_normed_bin_median = np.copy(all_bin_median)
    for i in range(len(categories)):
        rnn_normed_bin_mean[:, i] = rnn_normed_bin_mean[:, i] / max(rnn_normed_bin_mean[:, i])
        rnn_normed_bin_median[:, i] = rnn_normed_bin_median[:, i] / max(rnn_normed_bin_median[:, i])

        untrained_normed_bin_mean[:, i] = untrained_normed_bin_mean[:, i] / untrained_normed_bin_mean[0, i]
        untrained_normed_bin_median[:, i] = untrained_normed_bin_median[:, i] / untrained_normed_bin_median[0, i]

    df = {
        'mean': all_bin_mean.flatten(),
        'median': all_bin_median.flatten(),
        'ROARN-normalized': rnn_normed_bin_mean.flatten(),
        'ROARN-normalized median': rnn_normed_bin_median.flatten(),
        'Untrained-normalized': untrained_normed_bin_mean.flatten(),
        'Untrained-normalized median': untrained_normed_bin_median.flatten(),
        'mean_std': all_bin_mean_std.flatten(),
        'median_std': all_bin_median_std.flatten(),
        'model_type': np.array(all_model_types).flatten(),
        'bin_name': np.array(all_bin_id).flatten()
    }
    df = pd.DataFrame(df)
    for m in ['mean', 'median', 'ROARN-normalized', 'ROARN-normalized median', 'Untrained-normalized', 'Untrained-normalized median']:
        ax = sns.barplot(data=df, x='bin_name', y=m, hue='model_type', palette=sns.set_palette(sns.color_palette("deep")))
        if len(m) >= 6:
            plt.ylabel(f'{str(m).title()}\nNeural Predictivity')
        else:
            if m =='mean':
                plt.ylabel(f'Neural Predictivity')
            else:
                plt.ylabel(f'{str(m).title()} Neural Predictivity')
        ax.set(xlabel=None)

        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, -0.35), ncol=4, title=None, frameon=False)
        if save_legend:
            export_legend(ax, f"{save_fig_folder}/")
        ax.get_legend().remove()
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        if m == 'mean':
            er = 'mean_std'
        else:
            er = 'median_std'
        plt.errorbar(x=x_coords, y=y_coords, yerr=df[er], fmt="none", c="k", capsize=3)  # , elinewidth=0.5
        plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        if save_fig:
            os.makedirs(f"{save_fig_folder}/", exist_ok=True)
            fig_path = f"{save_fig_folder}/memory-barplot-{reg_name}-{m}-distance-percep_vs_mem{len(categories)}.png"
            plt.savefig(fig_path, bbox_inches='tight', dpi=900)
            plt.savefig(fig_path[:-3] + 'svg', bbox_inches='tight', format='svg')
            plt.clf()
            plt.close()
        else:
            plt.show()

    print('row-wise done')



if do_column_wise:
    ####################################################################################
    # For each neuron
    # 1. Load best regression hyperparameters for 1 step
    # 2. For column_index
    #   2.a) Fit regressor on 30 steps of this feature
    #   2.b) Get correlation score for that column
    # 3. Plot x = row index, y = correlation
    ####################################################################################
    regression_file = "regression-meta-2023-08-30-1693425148.5576088-memory-cc-v2_rnn_3_2_MiniGrid-Associative-MemoryS7RMTM-v0_1steps.txt"
    filename = folder_path + regression_file
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    previous_gridsearch_results = lines

    reg_name = 'LinearSVR'
    reg_type = LinearSVR

    reg = reg_type()
    normalize_input = False
    if not normalize_input:
        reg_model = reg
    else:
        reg_model = Pipeline([('scaler', StandardScaler()), ('reg', reg)])
    col_wise_corr_per_neuron = {}
    for neuron_id in metadata.keys():
        number_step = 30
        if aligning_with_activations:
            neuron_metadata, activations = align_metadata_and_activation(metadata, activations, neuron_id)
        else:
            neuron_metadata = metadata[neuron_id]
        X_metadata, firing_rates = get_meta_dataset(neuron_metadata, fr_index, factors_start_index,
                                                    add_trial_phases=add_trial_phases, n_steps=number_step)
        best_parameters = get_regression_params(previous_gridsearch_results, neuron_id, regression_model=reg_name)
        reg_model.set_params(**best_parameters)

        correlation_per_column = []
        nb_feature = int(X_metadata.shape[1] / number_step)
        assert len(feature_names) == nb_feature
        for column_index in range(nb_feature):
            column_through_time = [column_index + (i * nb_feature) for i in range(number_step)]
            col_i = X_metadata[:, column_through_time]
            predicted_ff, reg_model, kfold_weights = timeseries_crosval_predict(col_i, firing_rates, reg_model)
            corr, pvalue = pearsonr(predicted_ff, firing_rates)
            correlation_per_column.append(corr)
        folder_and_title_name = f"Column-wise ({reg_name})"
        correlation_per_column = np.array(correlation_per_column).reshape((nb_feature, 1))
        plot_matrix(correlation_per_column, feature_names, folder_and_title_name)
        col_wise_corr_per_neuron[neuron_id] = correlation_per_column

    with open(f"{save_fig_folder}/col_wise_profile.pickle", 'wb') as handle:
        pickle.dump(col_wise_corr_per_neuron, handle, protocol=pickle.HIGHEST_PROTOCOL)

