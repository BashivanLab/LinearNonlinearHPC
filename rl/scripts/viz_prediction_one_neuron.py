import os
import argparse
import ast
import time
from random import shuffle
from random import sample

import numpy
from scripts.create_metadata_matrix import align_metadata_and_activation
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn import linear_model
from scipy.stats import pearsonr, spearmanr
# from sklearn.model_selection import cross_val_predict

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model-type", default=None,
                    help="type of model to use (default from model.py)")
parser.add_argument("--location-type", default=9,
                    help="way to separate space in the maze. 2 ways: 5 directional locations or (default) 9 allocentric locations")
parser.add_argument("--exp-name", default=None,
                    help="Name of the experiment to identify the output files")
parser.add_argument("--neuron", default='W20150128_A0M1_3',
                    help="Neuron on which to perform the clustering analysis")
parser.add_argument("--type-regression-model", default='LinearSVR',
                    help="Regression model to use: Lasso, ElasticNet, LinearSVR, or Ridge")
parser.add_argument("--hyperparams-reg-file", required=True,
                    help="regression-output-[...].txt files that contain the hyperparameters to load in the regression model")
parser.add_argument("--hyperparams-location", default=None,
                    help="regression-output-[...].txt files that contain the hyperparameters to load in the location regression model")
parser.add_argument("--hyperparams-idealobs", default=None,
                    help="regression-output-[...].txt files that contain the hyperparameters to load in the ideal observer regression model")
parser.add_argument("--checkpoint", default=None,
                    help="Checkpoint id if want to select a specific ones. If none, select the final checkpoint")
args = parser.parse_args()

only_ff_and_meta = False

roarn_color = '#D7263D'
neuron_color = '#4F5669'


# indexes to select information from metadata.pickle
sample_index = 0
fr_index = 1
factors_start_index = 2

also_plot_location = args.hyperparams_location is not None
also_plot_ideal_observer = args.hyperparams_idealobs is not None
checkpoint_id = args.checkpoint
save_fig = True
reg_name = args.type_regression_model
reg_types = ['Lasso', 'ElasticNet', 'LinearSVR', 'Ridge']
red_models = [linear_model.Lasso, linear_model.ElasticNet, LinearSVR, linear_model.Ridge]
neuron_id = args.neuron
regression_file = args.hyperparams_reg_file
experiment_name = f"{args.model}"
if args.exp_name != '':
    experiment_name = f"{experiment_name}_{args.exp_name}"


####################################
# Load activations hdf5
####################################
# if checkpoint, path to checkpoint folder
if checkpoint_id is not None:
    h5_file_path = f'activations_{args.model}_{args.env}/activations_{args.model}_{args.env}_{checkpoint_id}.h5'
else:
    h5_file_path = f'activations_{args.model}_{args.env}.h5'

save_fig_folder = f"rl/figures/single_neuron_prediction_viz/{experiment_name}/{neuron_id}"
folder_path = ""

if args.model_type is not None and 'epn' in args.model_type:
    layer_name = 'max_layer'
else:
    layer_name = 'cell_state'

# Load activations on real trials
hf = h5py.File(h5_file_path, 'r')
hf_activations = hf[f"aligned_activations_{layer_name}"]

# Need session_setting to translate the metadata to labels
session_id = neuron_id.split('_')[0]
ind_session = np.where(hf['env_data']['session_id'][:] == session_id.encode('utf-8'))[0]
session_setting = hf['env_data']['purple_context_reward_values_worst_to_best'][ind_session[0]]
# hf.close()

# Load activations and firing rate from the real trials
def get_individual_variables(neuron_metadata):
    # TODO If change order or add variables, need to change this function.
    # If not, it will silently output non-sense features.
    # To be sure that it doesn't get misaligned,
    # Should change how synthetic matrix is created to automatically generate this information and save it with the matrix
    assert neuron_metadata.shape[1] == 27
    variables_start_end_indexes = {
        # end index is excluded
        'Location': (0, 9),
        'Direction': (9, 11),
        'Context': (11, 13),
        'Object_west': (13, 16),
        'Object_east': (16, 19),
        'Object_chosen': (19, 22),
        'Reward': (22, 25),
        'Choice_optimality': (25, 27),
    }
    individual_variables = {}
    for v in variables_start_end_indexes.keys():
        start_i, end_i = variables_start_end_indexes[v]
        individual_variables[v] = neuron_metadata[:, start_i:end_i]
    return individual_variables


def onehot_to_int(neuron_metadata):
    """
    Transform onehot encoding into an integer, e.g.:
    np.zeros (i.e. [0,0,0,0]) -> 0
    np.eye(3)[0] (i.e. [1,0,0]) -> 1
    np.eye(3)[2] (i.e. [0,0,1]) -> 3
    or generally. np.eye(3)[x] -> x+1
    """
    onehot_individual_variables = get_individual_variables(neuron_metadata)
    int_individual_variables = {}
    for v in onehot_individual_variables.keys():
        int_individual_variables[v] = []
        for onehot_row in onehot_individual_variables[v]:
            if all(onehot_row == 0):
                int_individual_variables[v].append(0)
            else:
                int_individual_variables[v].append(np.where(onehot_row)[0][0] + 1)
    return int_individual_variables


def onehot_to_label(neuron_metadata, session_setting):
    direction_int_to_label = {
        1: 'South',
        2: 'North'
    }
    context_int_to_label = {
        0: 'Not visible',
        1: 'Steel',
        2: 'Wood'
    }
    wood_setting = list(session_setting)[::-1]
    goal_int_to_label = {
        0: 'Not visible',
        1: wood_setting[0].decode('UTF-8'),
        2: wood_setting[1].decode('UTF-8'),
        3: wood_setting[2].decode('UTF-8')
    }
    reward_int_to_label = {
        0: 'No reward',
        1: 'Low',
        2: 'Medium',
        3: 'High'
    }
    optimality_int_to_label = {
        0: 'No choice',
        1: 'Incorrect',
        2: 'Correct',
    }
    transform_9_alloc_to_phases = {  # keys = (direction, 9_allocentric_location)
        ('North', 6): 'Context appearance',  # Corr S
        ('South', 6): 'Context appearance',  # Corr S
        ('North', 5): 'Context appearance',  # Center
        ('South', 5): 'Context appearance',  # Center
        ('North', 4): 'Context appearance',  # Corr N
        ('South', 4): 'Context appearance',  # Corr N

        ('South', 9): 'Object approach (Decision)',  # Goal SE
        ('South', 8): 'Object approach (Decision)',  # Goal SW
        ('South', 7): 'Object appearance',  # Dec S
        ('South', 3): 'Precontext',  # Dec N
        ('South', 2): 'Trial start',  # Goal NE
        ('South', 1): 'Trial start',  # Goal NW

        ('North', 9): 'Trial start',  # Goal SE
        ('North', 8): 'Trial start',  # Goal SW
        ('North', 7): 'Precontext',  # Dec S
        ('North', 3): 'Object appearance',  # Dec N
        ('North', 2): 'Object approach (Decision)',  # Goal NE
        ('North', 1): 'Object approach (Decision)',  # Goal NW
    }

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

    keep_location = {  # cool use maze part name instead
        9: 9,  # Goal SE
        8: 8,  # Goal SW
        7: 7,  # Dec S
        6: 6,  # Corr S
        5: 5,  # Center
        4: 4,  # Corr N
        3: 3,  # Dec N
        2: 2,  # Goal NE
        1: 1,  # Goal NW
    }
    label_transformation = {'Location': (keep_location, transform_9_alloc_to_phases, transform_9_alloc_to_5dir),
                            'Direction': direction_int_to_label,
                            'Context': context_int_to_label,
                            'Object_west': goal_int_to_label,
                            'Object_east': goal_int_to_label,
                            'Object_chosen': goal_int_to_label,
                            'Reward': reward_int_to_label,
                            'Choice_optimality': optimality_int_to_label
                            }
    int_individual_variables = onehot_to_int(neuron_metadata)
    label_individual_variables = {}
    for v in sorted(int_individual_variables.keys()):
        trans = label_transformation[v]
        if v == 'Location':
            label_individual_variables[v] = [trans[0][e] for e in int_individual_variables[v]]
            phase = []
            location_5dir = []
            for i in range(len(int_individual_variables[v])):
                direction_i = direction_int_to_label[int_individual_variables['Direction'][i]]
                phase.append(trans[1][(direction_i, int_individual_variables[v][i])])
                location_5dir.append(trans[2][(direction_i, int_individual_variables[v][i])])
            label_individual_variables['Trial_phase'] = phase
            label_individual_variables['Location_5dir'] = location_5dir
        else:
            label_individual_variables[v] = [trans[e] for e in int_individual_variables[v]]
    return label_individual_variables

def get_regression_params(lines, wanted_neuron, regression_model):
    for line in lines:
        neuron = line[8:line.find(", Regressor: ")]
        if wanted_neuron == neuron:
            reg_model = line[line.find("Regressor: ") + len("Regressor: "):line.find(", Best_result:")]
            if reg_model == regression_model:
                params_string = line[line.find("Param: ") + len("Param: "):line.find(", Runtime:")]
                params = ast.literal_eval(params_string)
                return params


# Load best regression hyperparameters for that neuron
def reg_model_with_loaded_hyperparam(regression_file):
    filename = folder_path + regression_file
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    previous_gridsearch_results = lines

    reg = red_models[reg_types.index(reg_name)]()
    best_parameters = get_regression_params(previous_gridsearch_results, neuron_id, regression_model=reg_name)
    normalize_input = 'reg' in list(best_parameters.keys())[0]
    if not normalize_input:
        reg_model = reg
    else:
        reg_model = Pipeline([('scaler', StandardScaler()), ('reg', reg)])  # reg #
    reg_model.set_params(**best_parameters)

    return reg_model

# Time series without multiple colors:

def plot_features_over_time(data, feature_names=None, save_fig_folder=None, fig_name=None):
    """
    Plots binary features over time for a numpy array of shape (timesteps, number of features).

    Parameters:
    data (numpy.ndarray): A 2D numpy array of shape (timesteps, number of features).
    feature_names (list of str): Optional list of feature names for labeling y-axis.
    """
    timesteps = data.shape[0]
    num_features = data.shape[1]

    # Create a figure and axes, adjusting size for visibility
    fig, ax = plt.subplots(num_features, 1, sharex=True, figsize=(10, num_features * 0.5))

    # If only one feature, ax will not be an array, so we need to make it iterable
    if num_features == 1:
        ax = [ax]

    # Plot each feature on its own row
    for i in range(num_features):
        ax[i].step(np.arange(timesteps), data[:, i], where='mid', color='black')
        ax[i].set_ylim(-0.1, 1.1)  # Binary features, so y should be between 0 and 1
        ax[i].set_yticks([])  # Hide y-axis ticks

        if feature_names is not None:
            # Set feature names as y-axis labels with reduced labelpad for less space
            ax[i].set_ylabel(f"{feature_names[i]}", rotation=0, labelpad=10, va='center', ha='right', fontsize=25)

        # Remove spines and ticks for a cleaner plot
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].tick_params(left=False, bottom=False)  # Hide ticks

    # Remove x-axis labels from all but the last plot
    for a in ax[:-1]:
        a.set_xticks([])  # Hide x-axis ticks

    # Set the x-axis label only for the bottom plot
    # ax[-1].set_xlabel('Time', fontsize=25)
    ax[-1].set_xlabel('Time', fontsize=25, ha='right', labelpad=20)
    # Adjust the x position to align it to the right
    ax[-1].xaxis.set_label_coords(0.98, -0.3)  # Move x label to the right side

    # Adjust the layout for readability and fitting of labels
    plt.tight_layout(pad=2.5)  # Reduce the padding slightly
    plt.subplots_adjust(left=0.35)  # Adjust left margin for better label fitting
    # plt.show()
    if save_fig_folder is not None:
        if fig_name is None:
            fig_name = 'plot_features_over_time'
        plt.savefig(f"{save_fig_folder}/{fig_name}", bbox_inches='tight', dpi=900)
        plt.savefig(f"{save_fig_folder}/svg/{fig_name}.svg", bbox_inches='tight', dpi=900,
                    format='svg')
    else:
        plt.show()
    plt.close()
    plt.clf()

def plot_float_features_over_time(data, feature_names=None, save_fig_folder=None, fig_name=None):
    """
    Plots float features over time for a numpy array of shape (timesteps, num_features).

    Parameters:
    data (numpy.ndarray): A 2D numpy array of shape (timesteps, num_features).
    feature_names (list of str): Optional list of feature names for labeling y-axis.
    """
    timesteps = data.shape[0]
    num_features = data.shape[1]

    # Create a figure and axes, adjusting size for visibility
    fig, ax = plt.subplots(num_features, 1, sharex=True, figsize=(10, num_features * 0.5))

    # If only one feature, ax will not be an array, so we need to make it iterable
    if num_features == 1:
        ax = [ax]

    # Plot each feature on its own row
    for i in range(num_features):
        ax[i].plot(np.arange(timesteps), data[:, i], color='black', linewidth=2)
        ax[i].set_ylim(np.min(data) - 1, np.max(data) + 1)  # Set y-limits based on data range
        ax[i].set_yticks([])  # Hide y-axis ticks

        if feature_names is not None:
            # Set feature names as y-axis labels
            ax[i].set_ylabel(f"{feature_names[i]}", rotation=0, labelpad=10, va='center', ha='right', fontsize=25)

        # Remove spines and ticks for a cleaner plot
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].tick_params(left=False, bottom=False)  # Hide ticks

    # Remove x-axis labels from all but the last plot
    for a in ax[:-1]:
        a.set_xticks([])  # Hide x-axis ticks

    # Set the x-axis label only for the bottom plot, positioned to the right
    ax[-1].set_xlabel('Time', fontsize=25, labelpad=20)
    ax[-1].xaxis.set_label_coords(0.98, -0.3)  # Move x label to the right side

    # Add a common y-axis label for all plots
    y_label_x_pos = 0.33 if feature_names is None else 0.04  # Same y position for both cases
    fig.text(y_label_x_pos, 0.5, 'Neural activity', ha='center', va='center', rotation='vertical', fontsize=25)

    # Adjust the layout for readability and fitting of labels
    plt.tight_layout(pad=2.5)  # Reduce the padding slightly
    plt.subplots_adjust(left=0.35)  # Adjust left margin for better label fitting
    # plt.show()
    if save_fig_folder is not None:
        if fig_name is None:
            fig_name = 'plot_float_features_over_time'
        plt.savefig(f"{save_fig_folder}/{fig_name}", bbox_inches='tight', dpi=900)
        plt.savefig(f"{save_fig_folder}/svg/{fig_name}.svg", bbox_inches='tight', dpi=900,
                    format='svg')
    else:
        plt.show()
    plt.close()
    plt.clf()



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


example_neurons = ['W20150304_P1M0_3', 'R20141212_Hc9_4', 'W20150209_A0M1_2',
       'R20141212_Hc9_3', 'W20150320_P2M2_3', 'W20150217_P1M2_2',
       'R20140910_Hc7_3', 'W20150319_P2M4_4', 'W20150316_P2M4_3',
       'W20150128_A0M1_3']

# Firing rate and meta
if only_ff_and_meta:

    add_trial_phases = True
    feature_names = ['North-West Arm', 'North-East Arm', 'North Branching', 'North Corridor', 'Corridor Center',
                     'South Corridor', 'South Branching',
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
            ['Trial start', 'Precontext', 'Context appearance', 'Object appearance', 'Object selection'])

    save_fig_folder = f"figures/single_neuron_prediction_viz/{experiment_name}_no_model"
    os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)

    sns.set_theme()
    sns.set_context("poster")
    sns.set_style("white")


    pickle_in = open('metadata.pickle', "rb")
    metadata = pickle.load(pickle_in)
    pickle_in.close()
    neuron_metadata = metadata[neuron_id]
    firing_rates = [e[fr_index] for e in neuron_metadata]
    neuron_metadata = [e[factors_start_index:] for e in neuron_metadata]  # Remove sample_id and firing rates
    if add_trial_phases:
        trial_phases = [get_trial_phases(e) for e in neuron_metadata] # Remove sample_id and firing rates
        neuron_metadata = np.concatenate((neuron_metadata, trial_phases), axis=1)


    # viz_start, viz_end = 0, len(firing_rates)
    viz_start = 1499
    viz_end = viz_start + 75
    plot_features_over_time(neuron_metadata[viz_start: viz_end, :], feature_names,
                            save_fig_folder=save_fig_folder,
                            fig_name=f"meta_{neuron_id}_{viz_start}-{viz_end}")
    n_ids = example_neurons
    multiple_neuron_ff = []
    for n in n_ids:
        tmp_neuron_metadata = metadata[n]
        firing_rates = [e[fr_index] for e in tmp_neuron_metadata]

        multiple_neuron_ff.append(firing_rates[viz_start:viz_end])
    multiple_neuron_ff = np.array(multiple_neuron_ff).T
    neuron_name_list = [f'Neuron {i}' for i in range(0, multiple_neuron_ff.shape[1]-1)]  # Neurons 1 to N-1
    neuron_name_list.append('Neuron N')
    plot_float_features_over_time(multiple_neuron_ff, feature_names=None, save_fig_folder=save_fig_folder,
                                  fig_name=f"ts_{'-'.join(n_ids)}__{viz_start}-{viz_end}")


for neuron_id in example_neurons:
    save_fig_folder = f"figures/single_neuron_prediction_viz/{experiment_name}/{neuron_id}"
    os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)


    def simple_timeseries(firing_rates, predicted_ff, viz_start, viz_end, second_predicted_ff=None, second_label='', colors=None):
        if colors is None:
            colors = ['black', 'blue']
        if not isinstance(colors, list):
            colors = [colors]
        plt.plot(firing_rates[viz_start:viz_end], label='Actual', color=colors[0])
        ax = plt.gca()
        if predicted_ff is not None:
            plt.plot(predicted_ff[viz_start:viz_end], label='Predicted', zorder=3, color=colors[1], alpha=0.8)
        if second_predicted_ff is not None:
            plt.plot(second_predicted_ff[viz_start:viz_end], label=second_label, zorder=3, color=[2], alpha=0.8)
        # plt.legend(loc='upper right')
        if predicted_ff is not None or second_predicted_ff is not None:
            plt.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Firing rates (Hz)')
        ax.set_xticks([])
        fig = plt.gcf()
        fig.set_size_inches(10, 4.8)
        plt.tight_layout()
        plt.title(f"{neuron_id[0] + neuron_id[5:].replace('_', '.')}")
        sns.despine()
        if save_fig:
            fig_name = f"ts-prediction-{viz_start}-{viz_end}"
            if second_predicted_ff is not None:
                fig_name = fig_name + '_with_' + second_label
            plt.savefig(f"{save_fig_folder}/{fig_name}", bbox_inches='tight', dpi=900)
            plt.savefig(f"{save_fig_folder}/svg/{fig_name}.svg", bbox_inches='tight', dpi=900,
                        format='svg')
        else:
            plt.show()
        plt.close()
        plt.clf()

    def plot_scatter(title, neuron_id, predicted_ff, firing_rates, save_fig, color=None):
        plt.scatter(firing_rates, predicted_ff, color=color)
        plt.title(f"{neuron_id[0] + neuron_id[5:].replace('_', '.')} {title}")
        fig = plt.gcf()
        max_ff = np.maximum(np.max(firing_rates), np.max(firing_rates))
        plt.plot([0, max_ff], [0, max_ff], '--', alpha=0.5, color='black')
        without_nan = (np.isnan(firing_rates) + np.isnan(predicted_ff)) == False
        corr, pvalue = pearsonr(np.array(firing_rates)[without_nan], np.array(predicted_ff)[without_nan])
        plt.annotate("$\it{r}$" + f" = {corr:.2f}", (0.01, 0.94), xycoords='axes fraction')
        plt.xlabel(f'Firing rates (Hz)')
        plt.ylabel(f'Predicted firing rates (Hz)')
        sns.despine()
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{save_fig_folder}/scatter-predictions-{title}", bbox_inches='tight',
                        dpi=900)
            plt.savefig(f"{save_fig_folder}/svg/scatter-predictions-{title}.svg", bbox_inches='tight',
                        dpi=900, format='svg')
            plt.clf()
            plt.close()
        else:
            plt.show()

    pickle_in = open('metadata.pickle', "rb")
    metadata = pickle.load(pickle_in)
    pickle_in.close()
    neuron_metadata, activations = align_metadata_and_activation(metadata, hf_activations, neuron_id)
    firing_rates = [e[fr_index] for e in neuron_metadata]
    neuron_metadata = [e[factors_start_index:] for e in neuron_metadata] # Remove sample_id and firing rates
    neuron_locations = [e[factors_start_index:9] for e in neuron_metadata] # Remove sample_id and firing rates
    neuron_metadata = np.array(neuron_metadata)
    metadata_labels = onehot_to_label(neuron_metadata, session_setting) # Used to identify the time series point
    metadata_labels_int = onehot_to_int(neuron_metadata) # Used to identify the time series point

    reg_model = reg_model_with_loaded_hyperparam(regression_file)
    if also_plot_location:
        location_reg_model = reg_model_with_loaded_hyperparam(args.hyperparams_location)
    if also_plot_ideal_observer:
        ideal_obs_reg_model = reg_model_with_loaded_hyperparam(args.hyperparams_idealobs)

    ###############################################
    ##### Time-dependent cross_val_predict
    # timeseries_crosval_predict
    ###############################################

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
        fold_size = int(len(y)/(3*nb_folds))
        set_seperated_in_chunk = [ind_list[i:i + fold_size] for i in range(0, len(ind_list), fold_size)]
        left_over_inds = []
        nb_extra_chunks = len(set_seperated_in_chunk) - 3*nb_folds
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
        indices = list(range(nb_folds*3)) # all indices that are not left_over
        # Get all index separated per third, and shuffle them
        test_chunk_indexes = [sample(indices[i:i+nb_folds], len(indices[i:i+nb_folds])) for i in range(0, len(indices), nb_folds)]
        predicted_ff = [0 for _ in y]
        for _ in range(nb_folds):
            test_ind = set_seperated_in_chunk[[e.pop() for e in test_chunk_indexes]].flatten()
            test_ind = np.append(test_ind, np.array(left_over_per_fold.pop(), dtype=np.int64)) # Add some left_over_inds in each folds until they are all tested/predicted
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

    predicted_ff, reg_model, _ = timeseries_crosval_predict(X=activations, y=firing_rates, reg_model=reg_model)
    if also_plot_location:
        location_predicted_ff, location_reg_model, _ = timeseries_crosval_predict(X=neuron_locations, y=firing_rates, reg_model=location_reg_model)
    if also_plot_ideal_observer:
        ideal_obs_predicted_ff, ideal_obs_reg_model, _ = timeseries_crosval_predict(X=neuron_metadata, y=firing_rates, reg_model=ideal_obs_reg_model)


    sns.set_theme()
    # sns.set(font="Verdana") # default = "sans-serif"
    sns.set_style("white")
    sns.set_context("notebook")
    sns.set_theme(context="talk", style='white', font_scale=1.3)
    plot_scatter("", neuron_id, predicted_ff, firing_rates, save_fig, color=neuron_color)
    if also_plot_location:
        plot_scatter("Location", neuron_id, location_predicted_ff, firing_rates, save_fig)
    if also_plot_ideal_observer:
        plot_scatter("Ideal observer", neuron_id, ideal_obs_predicted_ff, firing_rates, save_fig)


    sns.set_theme()
    sns.set_context("talk")
    sns.set_style("white")
    sns.set_theme(context="talk", style='white', font_scale=1.2)

    viz_start, viz_end = 0, 50
    while viz_end <= len(firing_rates):
        simple_timeseries(firing_rates, predicted_ff, viz_start, viz_end, colors=[neuron_color, roarn_color])
        viz_start, viz_end = viz_start + 50, viz_end + 50

    viz_start, viz_end = 0, len(firing_rates)
    simple_timeseries(firing_rates, predicted_ff, viz_start, viz_end, colors=[neuron_color, roarn_color])
    if also_plot_location:
        simple_timeseries(firing_rates, predicted_ff, viz_start, viz_end, second_predicted_ff=location_predicted_ff, second_label='Location')
        # simple_timeseries(firing_rates, location_predicted_ff, viz_start, viz_end)
    if also_plot_ideal_observer:
        # simple_timeseries(firing_rates, ideal_obs_predicted_ff, viz_start, viz_end)
        simple_timeseries(firing_rates, predicted_ff, viz_start, viz_end, second_predicted_ff=ideal_obs_predicted_ff, second_label='Ideal observer')


# for v in metadata_labels.keys():
v = "Location_5dir"
viz_start, viz_end = 0, len(firing_rates) #0, len(firing_rates) #0, 900 #100, 500
if v == 'Location_5dir':
    N = len(np.unique(metadata_labels[v][viz_start:viz_end]))
    tag = [e-1 for e in metadata_labels[v][viz_start:viz_end]]
else:
    N = len(np.unique(metadata_labels_int[v][viz_start:viz_end]))
    tag = [e-1 for e in metadata_labels_int[v][viz_start:viz_end]]

colors = [(0.054901960784313725, 0.803921568627451, 0.5529411764705883), (0, 0.2901960784313726, 0.7686274509803922)]# (0.17254901960784313, 0.6509803921568628, 0.49019607843137253) [(0, 0.7686274509803922, 0.2901960784313726), (0, 0.6, 0.7686274509803922)]  # first color is black, last is red
cmap = LinearSegmentedColormap.from_list("Custom", colors, N=N)
bounds = np.linspace(0, N, N + 1)    # define the bins and normalize
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
x = range(len(np.arange(viz_start, viz_end)))
points = np.array([x, firing_rates[viz_start:viz_end]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm, label='Real')
lc.set_array(tag)
lc.set_linewidth(2)
plt.gca().add_collection(lc)

ax = plt.gca()
plt.plot(predicted_ff[viz_start:viz_end], label='Predicted', zorder=3, color='black', alpha=0.8) # , alpha=0.7 linestyle='dashed', color='black'
plt.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Firing rates (Hz)')
ax.set_xticks([])
cb = plt.colorbar(lc, spacing='proportional', ticks=bounds)
cb.set_label(v)
fig = plt.gcf()
fig.set_size_inches(10, 4.8)
plt.tight_layout()
plt.title(f"{neuron_id[0] + neuron_id[5:].replace('_', '.')}")
sns.despine()
if save_fig:
    if viz_start != 0 or viz_end != len(firing_rates):
        plt.savefig(f"{save_fig_folder}/ts-prediction-{v}-labels-zoomed-{viz_start}-{viz_end}.png", bbox_inches='tight', dpi=400)
    else:
        plt.savefig(f"{save_fig_folder}/ts-prediction-{v}-labels.png", bbox_inches='tight', dpi=400)
    plt.clf()
    plt.close()
else:
    plt.show()

print('.')