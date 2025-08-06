import os
from collections import Counter
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
from scipy.stats import permutation_test
from scipy.stats import sem
from itertools import combinations
import pandas as pd
import pickle
import argparse
import h5py
import copy
import numpy as np
from gym_minigrid.envs.associativememory import *
from torch_ac.utils.penv import ParallelEnv
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model-type", default=None,
                    help="type of model to use (default from model.py)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--exp-name", default=None,
                    help="Name of the experiment to identify the output files")
parser.add_argument("--partition-type", default=9,
                    help="Either partitioning the maze into 9 or 5 locations")
parser.add_argument("--checkpoint", default=None,
                    help="Checkpoint id if want to select a specific ones. If none, select the final checkpoint")
parser.add_argument("--use-abs-activation", action='store_true',
                    help="Whether use absolute unit activation or not")
args = parser.parse_args()


spatially_tuned_color = '#4C72B0' #sns.color_palette('deep')[0] # #4C72B0 in hexadecimal
not_spatially_tuned_color = '#DD8452' #sns.color_palette('deep')[1]

spatial_colors = [spatially_tuned_color, not_spatially_tuned_color]
neuron_color = '#4F5669'
darker_neuron_color = (0.322, 0.322, 0.322) # for some plots, use a darkger gray
untrained_color = '#D17577' # '#E05265'
position_color = sns.color_palette('deep')[4] #'#8172B3'
io_color =  sns.color_palette('deep')[2] #'#55A868'
roarn_color = '#D7263D'
io_mem_color = (0.208, 0.51,0.298)
linear_color = '#007A3D'
untrained_linear_color = '#00A352'
model_colors = [untrained_color, position_color, io_color, linear_color, roarn_color]
perceptual_color = '#008381' # '#00bda3'
near_color = '#ab84a9' #56CBF9'
distant_color = '#a14da0'# '#00A4EB'
temporal_colors = [perceptual_color, near_color, distant_color]

use_abs = args.use_abs_activation
experiment_name = args.exp_name
save_fig = True
save_fig_folder = f"figures/spatial_neurons/{experiment_name}"


# HDF5 files are only needed to plot activation maps
checkpoint_id = args.checkpoint if args.checkpoint is None else int(args.checkpoint)
if checkpoint_id is not None:
    h5_file_path = f'activations_{args.model}_{args.env}/activations_{args.model}_{args.env}_{checkpoint_id}.h5'
else:
    h5_file_path = f'activations_{args.model}_{args.env}.h5'
untrained_h5_file_path = f'activations_{args.model}_{args.env}/activations_{args.model}_{args.env}_0.h5'
linear_h5_file_path = f'activations_{args.model.replace("rnn", "linear_rnn")}_{args.env}/activations_{args.model.replace("rnn", "linear_rnn")}_{args.env}_344.h5'
os.makedirs(f"{save_fig_folder}/", exist_ok=True)

############################################
# Plots activation maps
############################################
def build_neurons_activation_per_location(activation_per_trial, agent_positions_per_trial):
    """
    Build dictionary
    for each neuron
        for each location
            [each trial]
    :param activation_per_trial:
    :param agent_positions_per_trial: list where length = number of trials. Each element of the list = np.array of location e.g. agent_positions_per_trial[0] = [[1,4], [1,4], [1,4], [2,4], [3,4], ...]
    :return:
    neurons = {'neuron0': {(1,1): [activation_trial0, activation_trial1, ...],
                      (1,2): [activation_trial0, activation_trial1, ...],
                      ...,
                      }
          'neuron1': {(1,1): [activation_trial0, activation_trial1, ...],
                      (1,2): [activation_trial0, activation_trial1, ...],
                      ...,
                      },
          ...
          'neuron255': {(1,1): [activation_trial0, activation_trial1, ...],
                        (1,2): [activation_trial0, activation_trial1, ...],
                        ...,
                        },
          }
    """
    neurons = {}
    for n in range(activation_per_trial[0].shape[-1]):
        neurons['neuron' + str(n)] = {}  # initialization

    for trial in range(len(agent_positions_per_trial)):
        visited_location = set()
        duplicate_activation = {}
        for e in zip(agent_positions_per_trial[trial], activation_per_trial[trial]):
            location = tuple(e[0])
            if 'Associative-MemoryS9' in args.env and location in [(1, 1), (1, 2), (1, 6), (
            1, 7)]:
                continue
            if 'Associative-MemoryS7' in args.env and location in [(1, 1), (1, 5), (5, 1), (
            5, 5)]:
                continue
            if location in visited_location:
                duplicate_activation[location] = {}
            else:
                visited_location.add(location)
            if e[1].shape[0] == 1 or len(e[1].shape) == 1:
                neurons_activation = e[1].squeeze()
            else:
                # if want to parallelize envs, need to look if it correctly aligns activation and data between parallel envs
                raise NotImplementedError

            for n in range(len(neurons_activation)):  # Add activation at current location & trial  for each neuron
                if location in duplicate_activation.keys():
                    # Check if multiple activations for the same location in the same trial
                    duplicate_activation[location].setdefault('neuron' + str(n), []).append(neurons_activation[n])
                else:
                    neurons['neuron' + str(n)].setdefault(location, []).append(neurons_activation[n])

        for loc in duplicate_activation.keys():
            for n in range(len(neurons_activation)):
                # replace duplicates activation at the same location for the average activation
                neurons['neuron' + str(n)][loc][-1] = sum(duplicate_activation[loc]['neuron' + str(n)]) / len(
                    duplicate_activation[loc]['neuron' + str(n)])
    return neurons

def get_activation_per_location(env_data, activations, inspected_layer):
    index_start_trial = []  # index where each trial start
    for i in range(1, env_data['trial_id'][:].shape[0]):
        if env_data['trial_id'][i] != env_data['trial_id'][i - 1]:
            index_start_trial.append(i)

    agent_positions_per_trial = []
    activation_per_trial = []
    for i in range(len(index_start_trial[:-1])):  # ignore last trial as it's probably incomplete
        start = index_start_trial[i]  # start = index_start_trial[i]
        end = index_start_trial[
                  i + 1] - 1  # end = index_start_trial[i+1] # index for the end of the i-th trial which is the element before i+1
        agent_positions_per_trial.append(env_data['agent_location'][start:end, :])
        activation_per_trial.append(activations[inspected_layer][start:end])

    units = build_neurons_activation_per_location(activation_per_trial, agent_positions_per_trial)
    return units

def get_env_for_visualization(env=None, remove_distal_cue=False):
    if env is not None:
        env_viz = copy.deepcopy(env.envs[0])  # copy the env to modify it for each visualization
    # deepcopy was giving some error on CC, so recreate an env from scratch instead to get around that
    envs = []
    env = utils.make_env(args.env, args.seed + 10000 * 0)
    envs.append(env)
    env = ParallelEnv(envs)
    env_viz = env.envs[0]

    if 'Associative-Memory' in args.env:
        # Remove agent, goal, cue, and doors from grid
        y_pos_sidestep = 1
        if env_viz.goal_obj_down.cur_pos is not None:
            env_viz.grid.set(env_viz.goal_obj_down.cur_pos[0], env_viz.goal_obj_down.cur_pos[1] + y_pos_sidestep,
                             Wall())
            env_viz.grid.set(*env_viz.goal_obj_down.cur_pos, None)  # remove goal object down
            env_viz.grid.set(*env_viz.goal_obj_down.cur_pos, None)
        if env_viz.goal_obj_up.cur_pos is not None:
            env_viz.grid.set(*env_viz.goal_obj_up.cur_pos, None)  # remove goal object up
            env_viz.grid.set(env_viz.goal_obj_up.cur_pos[0], env_viz.goal_obj_up.cur_pos[1] - y_pos_sidestep, Wall())
            env_viz.grid.set(*env_viz.goal_obj_up.cur_pos, None)
        if env_viz.cue_obj.cur_pos is not None:
            env_viz.grid.set(*env_viz.cue_obj.cur_pos, Wall())  # remove cue object up
            env_viz.grid.set(env_viz.cue_obj.cur_pos[0], env_viz.cue_obj.cur_pos[1] + 2, Wall())  # remove cue object down
            env_viz.cue_obj.cur_pos = None
        if remove_distal_cue:
            upper_room_wall = env_viz.height // 2 - 2
            lower_room_wall = env_viz.height // 2 + 2
            env_viz.grid.set(*(env_viz.height // 2, upper_room_wall), Wall())
            env_viz.grid.set(*(env_viz.height // 2, lower_room_wall), Wall())

        env_viz.grid.set(*env_viz.agent_pos, None)  # remove agent

    return env_viz

Locminigrid = {
    9: (1, 4),  # Goal SE
    8: (1, 2),  # Goal SW
    7: (1, 3),  # Dec S
    6: (2, 3),  # Corr S
    5: (3, 3),  # Center
    4: (4, 3),  # Corr N
    3: (5, 3),  # Dec N
    2: (5, 4),  # Goal NE
    1: (5, 2),  # Goal NW
}

def plot_activity_over_maze(neurons, name_selected_neurons=None, folder_title='', use_absolute=False):
    if name_selected_neurons is None:
        name_selected_neurons = list(neurons.keys())

    cmap = plt.get_cmap('viridis')
    avg_units = {}  # average activation per location for each neuron
    for n in name_selected_neurons:
        max_activation = 0
        min_activation = 0
        avg_units[n] = {}
        env_viz = get_env_for_visualization(remove_distal_cue=True)

        for loc in neurons[n].keys():
            # average activation
            if use_absolute:
                # absolute when use tanh as activation function which allows both neg and pos values
                avg_units[n][loc] = np.mean(np.abs(neurons[n][loc]))
            else:
                avg_units[n][loc] = np.mean(neurons[n][loc])
            if avg_units[n][loc] > max_activation:
                max_activation = avg_units[n][loc]
            if avg_units[n][loc] < min_activation:
                min_activation = avg_units[n][loc]

        for loc in neurons[n].keys():
            # Normalize to determine the color from a gradient
            if max_activation == 0.0 and min_activation == 0.0:
                avg_units[n][loc] = avg_units[n][loc]
            else:
                avg_units[n][loc] = (avg_units[n][loc] - min_activation) / (
                        max_activation - min_activation)
            rgb = cmap(avg_units[n][loc], bytes=True)[
                  :3]  # get the color in the gradient corresponding to the activation
            if isinstance(loc, int):
                loc = Locminigrid[loc]
            env_viz.put_obj(Square(color=np.array(rgb)), loc[0], loc[1])  # color the location

        if save_fig:
            plt.axis('off')
            os.makedirs(f"{save_fig_folder}/activation_map_{folder_title}/", exist_ok=True)
            im = plt.imshow(np.rot90(env_viz.grid.render(tile_size=TILE_PIXELS)), interpolation='bilinear')
            plt.clim(min_activation, max_activation)
            cbar = plt.colorbar(im, ax=im.axes)
            plt.savefig(f"{save_fig_folder}/activation_map_{folder_title}/{n}")
            os.makedirs(f"{save_fig_folder}/activation_map_{folder_title}/svg", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/activation_map_{folder_title}/svg/{n}.svg", format='svg', bbox_inches='tight')
            plt.clf()
            plt.close()
        else:
            plt.show()

# Get firing rates per location for each neurons
pickle_in = open('metadata.pickle', "rb")
metadata = pickle.load(pickle_in)  # data from monkey experiment
pickle_in.close()
# indexes to select information from metadata.pickle
sample_index = 0
fr_index = 1
factors_start_index = 2
is_incorrect_index = 26
is_correct_index = 27

neurons = {}
for neuron_name in metadata.keys():
    if neuron_name not in neurons:
        neurons[neuron_name] = {}
        for loc in range(1, args.partition_type + 1):
            neurons[neuron_name][loc] = []
    for step in metadata[neuron_name]:
        location = step[0][2]
        firing_rate = step[fr_index]
        neurons[neuron_name][location].append(firing_rate)

if args.model_type is not None and 'epn' in args.model_type:
    inspected_layer = 'max'
    layer_name = 'max'
elif args.model_type == 'linear_rnn' or args.model_type == 'hidden_state':
    inspected_layer = 'memory_rnn_hidden_state'
    layer_name = 'hidden_state'
else:
    inspected_layer = 'memory_rnn_cell_state'
    layer_name = 'cell_state'
    linear_inspected_layer = 'memory_rnn_hidden_state'
    linear_layer_name = 'hidden_state'

hf = h5py.File(h5_file_path, 'r')
activations = hf['layer_activations']
env_data = hf['env_data']

# untrained_activations = None
# untrained_env_data = None
untrained_hf = h5py.File(untrained_h5_file_path, 'r')
untrained_activations = untrained_hf['layer_activations']
untrained_env_data = untrained_hf['env_data']

linear_hf = h5py.File(linear_h5_file_path, 'r')
linear_activations = linear_hf['layer_activations']
linear_env_data = linear_hf['env_data']



# Load 3 seeds pickles (if doesn't exist create it)
dir_path = "rl/figures/spatial_neurons/9nov2024/"
seed_files = [[dir_path+"memory-cc-v2_rnn_3_6/significant_untrained_linear_units_0.pickle",
               dir_path+"memory-cc-v2_rnn_3_6/significant_linear_units_344.pickle",
               dir_path+"memory-cc-v2_rnn_3_6/significant_untrained_units_0.pickle",
               dir_path+"memory-cc-v2_rnn_3_6/significant_units_383.pickle",
               dir_path+"memory-cc-v2_rnn_3_6/significant_neurons.pickle",
            ],
          [dir_path + "memory-cc-v2_rnn_3_5/significant_untrained_linear_units_0.pickle",
           dir_path + "memory-cc-v2_rnn_3_5/significant_linear_units_344.pickle",
           dir_path + "memory-cc-v2_rnn_3_5/significant_untrained_units_0.pickle",
           dir_path + "memory-cc-v2_rnn_3_5/significant_units_343.pickle",
           dir_path + "memory-cc-v2_rnn_3_5/significant_neurons.pickle",
           ],
          [dir_path + "memory-cc-v2_rnn_3_4/significant_untrained_linear_units_0.pickle",
           dir_path + "memory-cc-v2_rnn_3_4/significant_linear_units_344.pickle",
           dir_path + "memory-cc-v2_rnn_3_4/significant_untrained_units_0.pickle",
           dir_path + "memory-cc-v2_rnn_3_4/significant_units_344.pickle",
           dir_path + "memory-cc-v2_rnn_3_4/significant_neurons.pickle",
            ],
        ]
names = ['Linear untrained', 'Linear trained', 'Untrained', 'Trained', 'Neurons']
nb_seeds = len(seed_files)
significant_untrained_linear_units = []
significant_linear_units = []
significant_untrained_units = []
significant_units = []
significant_neurons = []
if not os.path.exists(seed_files[0][0]):
    # only create files for one seed
    # Create pickle files
    os.makedirs(f"{save_fig_folder}/{args.model}", exist_ok=True)
    pvalue_threshold = 0.01

    def statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    def statistic_abs(x, y, axis):
        # With artificial units, there are negative values
        #  and significantly lower values could be equivalent to real neurons significantly elevated fire rates
        return np.mean(np.abs(x), axis=axis) - np.mean(np.abs(y), axis=axis)

    def get_significant_neurons_locations(neurons, number_permutation=100, test_statistic=statistic,
                                          test_side='greater', pvalue=0.01):
        significant_neurons = {}
        pvalues = []
        for n in neurons.keys():
            significant_neurons[n] = {}
            # Compare if firing rates at location 'loc' are significantly higher than those at all other locations
            for loc in neurons[n].keys():
                tested_firing_rates = neurons[n][loc]
                control_locations = set(neurons[n].keys())
                control_locations.remove(loc)
                control_firing_rates = []
                for l in control_locations:
                    control_firing_rates.extend(neurons[n][l])

                test_result = permutation_test((tested_firing_rates, control_firing_rates), test_statistic,
                                               permutation_type='independent', vectorized=True,
                                               n_resamples=number_permutation,
                                               alternative=test_side)
                significant_neurons[n][loc] = test_result.pvalue <= pvalue
                pvalues.append(test_result.pvalue)

        # FDR correction
        fdr_pvalues = stats.false_discovery_control(pvalues)

        i = 0
        diff = []
        for n in significant_neurons.keys():
            for loc in significant_neurons[n].keys():
                if significant_neurons[n][loc] != (fdr_pvalues[i] <= pvalue):
                    diff.append((n, loc, pvalues[i], fdr_pvalues[i]))
                significant_neurons[n][loc] = fdr_pvalues[i] <= pvalue
                i += 1

        return significant_neurons


    significant_neurons = get_significant_neurons_locations(neurons, number_permutation=1000,
                                                            test_statistic=statistic, test_side='greater',
                                                            pvalue=pvalue_threshold)
    with open(f"{save_fig_folder}/{args.model}/significant_neurons.pickle", 'wb') as handle:
        pickle.dump(significant_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)

    units = get_activation_per_location(env_data, activations, inspected_layer)
    significant_units = get_significant_neurons_locations(units, number_permutation=1000,
                                                          test_statistic=statistic_abs, test_side='greater',
                                                          pvalue=pvalue_threshold)
    with open(f"{save_fig_folder}/{args.model}/significant_units_{checkpoint_id}.pickle", 'wb') as handle:
        pickle.dump(significant_units, handle, protocol=pickle.HIGHEST_PROTOCOL)

    untrained_units = get_activation_per_location(untrained_env_data, untrained_activations, inspected_layer)
    significant_untrained_units = get_significant_neurons_locations(untrained_units, number_permutation=1000,
                                                                    test_statistic=statistic_abs, test_side='greater',
                                                                    pvalue=pvalue_threshold)
    """
    Save significant_untrained_units, significant_units, significant_neurons into pickle files
    """
    with open(f"{save_fig_folder}/{args.model}/significant_untrained_units_0.pickle", 'wb') as handle:
        pickle.dump(significant_untrained_units, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    for seed_file in seed_files:
        with open(seed_file[0], 'rb') as handle:
            significant_untrained_linear_units.append(pickle.load(handle))
        with open(seed_file[1], 'rb') as handle:
            significant_linear_units.append(pickle.load(handle))
        with open(seed_file[2], 'rb') as handle:
            significant_untrained_units.append(pickle.load(handle))
        with open(seed_file[3], 'rb') as handle:
            significant_units.append(pickle.load(handle))
        with open(seed_file[4], 'rb') as handle:
            significant_neurons.append(pickle.load(handle))
significant_lists = [significant_untrained_linear_units, significant_linear_units, significant_untrained_units, significant_units, significant_neurons]



do_plot_activity_over_maze = False
if do_plot_activity_over_maze:
    units = get_activation_per_location(env_data, activations, inspected_layer)
    untrained_units = get_activation_per_location(untrained_env_data, untrained_activations, inspected_layer)
    linear_units = get_activation_per_location(linear_env_data, linear_activations, linear_inspected_layer)
    plot_activity_over_maze(neurons, name_selected_neurons=None, folder_title='neurons', use_absolute=False)
    plot_activity_over_maze(units, name_selected_neurons=None, folder_title=f'units/{args.model}', use_absolute=use_abs)
    plot_activity_over_maze(untrained_units, name_selected_neurons=None, folder_title=f'untrained_units/{args.model}',
                            use_absolute=use_abs)
    plot_activity_over_maze(linear_units, name_selected_neurons=None, folder_title=f'linear_units/{args.model}',
                            use_absolute=use_abs)
linear_hf.close()
untrained_hf.close()
hf.close()

if False:
    sns.set_theme(context="talk", style='white', font_scale=1.2)
    # Visualize units activations
    for i in range(0, 55):
        # i = 12
        viz_start = 0
        # viz_end = activations[inspected_layer].shape[0]
        viz_end = 75
        x = activations[inspected_layer][viz_start:viz_end,i]
        plt.plot(x, color='black', linewidth=3)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        fig = plt.gcf()
        fig.set_size_inches(10, 4.8)
        sns.despine()
        plt.tight_layout()
        os.makedirs(f"{save_fig_folder}/svg/timeseries", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/svg/timeseries/timeseries-unit{i}-{viz_start}-{viz_end}.svg", bbox_inches='tight', dpi=900,
                    format='svg')
        plt.close()
        plt.clf()

############################################
# Plots fig2b (coincident matrices)
############################################
def plot_coincident_matrix(data, location_labels, title, file_title):
    column_labels = location_labels
    row_labels = location_labels
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=cmap)
    ax.set_aspect(1) # set cells to be square
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.set_xticklabels(row_labels, minor=False, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(column_labels, minor=False)
    plt.xticks(rotation=45)
    cbar = plt.colorbar(heatmap, label='Probability')
    cbar.ax.locator_params(nbins=4)
    cbar.update_ticks()
    plt.xlabel('Additional locations')
    plt.ylabel('Reference locations')
    plt.title(f"Coincident Locations")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    if save_fig:
        os.makedirs(f"{save_fig_folder}", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/fig2b_{file_title}", bbox_inches='tight', dpi=900)
        os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/svg/fig2b_{file_title}.svg", bbox_inches='tight', dpi=900, format='svg')
        plt.clf()
        plt.close()
    else:
        plt.show()

def plot_fig2b(significant_neurons, locations, location_labels, file_title, title='Neurons', save_fig=True):
    data = np.zeros(len(locations) * len(locations)).reshape((len(locations), len(locations)))
    for loc in locations:
        # Select every neurons that are highly activated at loc and at another location
        preselection = []
        for n in significant_neurons.keys():
            if significant_neurons[n][loc] and len(
                    [(loc, is_high) for loc, is_high in significant_neurons[n].items() if is_high == True]) >= 2:
                preselection.append(n)
        # Compute Number of neurons that also reacts in the second location divided by total number of neurons that react in the primary location
        total = len(preselection)
        for second_loc in locations:
            for n in preselection:
                if significant_neurons[n][second_loc]:
                    data[locations.index(loc)][locations.index(second_loc)] += 1
            data[locations.index(loc)][locations.index(second_loc)] /= total
    if save_fig:
        plot_coincident_matrix(data, location_labels, title, file_title)
    return data

def remove_diag(x):
    x_no_diag = x.flatten()
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    return x_no_diag

minigrid_loc_labels = {
    (1, 4): 'Goal SE',
    (1, 2): 'Goal SW',
    (1, 3): 'Dec S',
    (2, 3): 'Corr S',
    (3, 3): 'Center',
    (4, 3): 'Corr N',
    (5, 3): 'Dec N',
    (5, 4): 'Goal NE',
    (5, 2):'Goal NW',
}

neuron_loc_labels = {
    9: 'Goal SE',
    8: 'Goal SW',
    7: 'Dec S',
    6: 'Corr S',
    5: 'Center',
    4: 'Corr N',
    3: 'Dec N',
    2: 'Goal NE',
    1: 'Goal NW',
}

# Average grid across the seeds. Is there a better way to also report the variance?
neuron_locations = list(significant_neurons[0][list(significant_neurons[0].keys())[0]].keys())
minigrid_locations = [(5, 2), (5, 4), (5, 3), (4, 3), (3, 3), (2, 3), (1, 3), (1, 2), (1, 4)]
location_labels = [minigrid_loc_labels[e] for e in minigrid_locations]
avg_grids = [np.zeros(len(minigrid_locations) * len(minigrid_locations)).reshape((len(minigrid_locations), len(minigrid_locations))) for s in range(len(names))]
corr_per_seeds = [[] for s in range(len(names))]
pvalue_per_seeds = [[] for s in range(len(names))]
for s in range(nb_seeds):
    for name_i in range(len(names)):
        if names[name_i] == 'Neurons':
            locations = neuron_locations
            title_name = names[name_i]
        else:
            locations = minigrid_locations
            title_name = f"{names[name_i]} units"
        # Get matrix for that seed and model
        grid = plot_fig2b(significant_lists[name_i][s], locations, location_labels, file_title=f"{names[name_i]} seed {s}", title=title_name, save_fig=False)
        # Accumulate it with other seeds of the model
        avg_grids[name_i] += grid
        # Accumulate correlations to report the average correlation with std
        neuron_grid = plot_fig2b(significant_lists[-1][s], neuron_locations, location_labels, file_title=f"{names[2]} seed {s}", save_fig=False)
        # With or without diagonal? I guess without because we don't actually care about the diagonal
        corr, pvalue = pearsonr(remove_diag(grid), remove_diag(neuron_grid))
        corr_per_seeds[name_i].append(corr)
        pvalue_per_seeds[name_i].append(pvalue)

#  Divide sum of grids by number of seeds to get average coincident matrices
avg_grids = [e/nb_seeds for e in avg_grids]
sns.set_theme(context="talk", style='white', font_scale=1.25)
for name_i in range(len(avg_grids)):
    if names[name_i] == 'Neurons':
        title_name = 'Neuron'
    else:
        title_name = f"Unit"
    #FIGURE MIXED SELECTIVITY
    plot_coincident_matrix(avg_grids[name_i], location_labels, title_name, f"avg_{names[name_i]}")

print(f'Neurons - Untrained linear units correlation: {np.mean(corr_per_seeds[0]).round(3)} ± {np.std(corr_per_seeds[0]).round(3)};   pvalue={pvalue_per_seeds[0]}')
print(f'Neurons - Trained linear units correlation: {np.mean(corr_per_seeds[1]).round(3)} ± {np.std(corr_per_seeds[1]).round(3)};   pvalue={pvalue_per_seeds[1]}')
print(f'Neurons - Untrained units correlation: {np.mean(corr_per_seeds[2]).round(3)} ± {np.std(corr_per_seeds[0]).round(3)};   pvalue={pvalue_per_seeds[0]}')
print(f'Neurons - Trained units correlation: {np.mean(corr_per_seeds[3]).round(3)} ± {np.std(corr_per_seeds[1]).round(3)};   pvalue={pvalue_per_seeds[1]}')


############################################
# Grouped bar plot where y-axis = proportion of neurons/units (%);
# bin = ['Spatially-tuned', 'Spatially-tuned at\ntask-equivalent locations'];
# group = ['Untrained', 'Trained', 'Neurons']
############################################
def grouped_bar_plot(data, group_names, bin_names, error_type, xlabel, ylabel, filename, folder_name=None,
                     keep_legend=False, averaging_axis=1, rotate_xticks=True, move_legend=None, colors=None, yticks=None,
                     add_scatterpoints=False):
    # Data: list with shape (nb_group, number of neurons in this group, number of bins)
    # if averaging_axis=1, average across dimension 1 of data (or across dimension averaging_axis-1 of data[i])
    # if averaging_axis=2, data shape = (nb_group, number of bins, number of neurons in this combination of group and bin)
    # If error_type is None, Doesn't average the second dimension of Data. (so works with 2D data)
    if colors is None:
        colors = sns.set_palette(sns.color_palette("deep"))
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
        ax = sns.barplot(data=df, x='bin_name', y=m, hue='neuron_type', palette=colors)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        # plt.ylim(0.0, 1.0)
        if not keep_legend:
            sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, -0.35), ncol=4, title=None, frameon=False)
            # if save_legend:
            #     export_legend(ax, filename, folder_name)
            ax.get_legend().remove()
        else:
            if move_legend is None:
                ax.legend().set_title('')
            elif move_legend == 'right':
                # Shrink current axis by 10%
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
                # Put a legend to the right of the current axis
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            elif move_legend == 'below':
                # Shrink current axis's height by 10% on the bottom
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.9])
                # Put a legend below current axis
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                          fancybox=True, shadow=True, ncol=5)

        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        if error_type is not None:
            plt.errorbar(x=x_coords, y=y_coords, yerr=df[error_type], fmt="none", c="k", capsize=3)  # , elinewidth=0.5
        if add_scatterpoints:
            for i, group_data in enumerate(data):
                for j, bin_data in enumerate(group_data.T):
                    x_val = j + (i * len(bin_names))  # Adjusts x position for groups; tweak for visual clarity
                    # jitter = 0.05 * (np.random.rand(len(bin_data)) - 0.5)  # Random jitter for visual separation
                    # ax.scatter(np.full(len(bin_data), x_val), bin_data, color=colors[i], alpha=0.6, edgecolor='k', linewidth=0.5)
                    ax.scatter([x_coords[j + (i * len(bin_names))]]*len(bin_data), bin_data, color='black', s=20, alpha=0.6)
                               #alpha=0.6, edgecolor='k', linewidth=0.5)
        if rotate_xticks:
            plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
        if yticks is not None:
            plt.yticks(yticks)
        plt.tight_layout()
        sns.despine(top=True, right=True)
        if save_fig:
            if folder_name is not None:
                os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}_{m}", bbox_inches='tight', dpi=900)
                os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/svg/{filename}_{m}.svg", bbox_inches='tight', dpi=900, format='svg')
            else:
                plt.savefig(f"{save_fig_folder}/{filename}_{m}", bbox_inches='tight', dpi=900)
                plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', format='svg')
            plt.clf()
            plt.close()
        else:
            plt.show()

def get_selectivity_patterns(significant_neurons, minigrid_locations=True):
    if minigrid_locations:
        loc_dict = Locminigrid
    else:
        loc_dict = {9: 9, 8: 8, 7: 7, 6: 6, 5: 5, 4: 4, 3: 3, 2: 2, 1: 1}
    spatial_selectivity_info = {}
    sym_spatial_selectivity_info = {}
    exclusive_sym_spatial_selectivity_info = {}
    for n in significant_neurons.keys():
        spatial_selectivity_info[n] = np.array(list(significant_neurons[n].values())).any()
        only_one = np.array(list(significant_neurons[n].values())).sum() == 1
        only_two = np.array(list(significant_neurons[n].values())).sum() == 2
        only_three = np.array(list(significant_neurons[n].values())).sum() == 3
        only_four = np.array(list(significant_neurons[n].values())).sum() == 4
        sym_spatial_selectivity_info[n] = {
            '1,9': significant_neurons[n][loc_dict[1]] and significant_neurons[n][loc_dict[9]],
            '2,8': significant_neurons[n][loc_dict[2]] and significant_neurons[n][loc_dict[8]],
            '1,2,8,9': significant_neurons[n][loc_dict[1]] and significant_neurons[n][loc_dict[9]] and
                       significant_neurons[n][loc_dict[2]] and significant_neurons[n][loc_dict[8]],
            '3,7': significant_neurons[n][loc_dict[3]] and significant_neurons[n][loc_dict[7]],
            '4,6': significant_neurons[n][loc_dict[4]] and significant_neurons[n][loc_dict[6]],
            '4,5,6': significant_neurons[n][loc_dict[4]] and significant_neurons[n][loc_dict[6]] and
                     significant_neurons[n][loc_dict[5]], # '5': significant_neurons[n][loc_dict[5]],
            }

        exclusive_sym_spatial_selectivity_info[n] = {
            '1,9': significant_neurons[n][loc_dict[1]] and significant_neurons[n][loc_dict[9]] and only_two,
            '2,8': significant_neurons[n][loc_dict[2]] and significant_neurons[n][loc_dict[8]] and only_two,
            '1,2,8,9': significant_neurons[n][loc_dict[1]] and significant_neurons[n][loc_dict[9]] and
                       significant_neurons[n][loc_dict[2]] and significant_neurons[n][loc_dict[8]] and only_four,
            '3,7': significant_neurons[n][loc_dict[3]] and significant_neurons[n][loc_dict[7]] and only_two,
            '4,6': significant_neurons[n][loc_dict[4]] and significant_neurons[n][loc_dict[6]] and only_two,
            '4,5,6': significant_neurons[n][loc_dict[4]] and significant_neurons[n][loc_dict[6]] and
                     significant_neurons[n][loc_dict[5]] and only_three, # '5': significant_neurons[n][loc_dict[5]] and only_one,
        }
    return spatial_selectivity_info, sym_spatial_selectivity_info, exclusive_sym_spatial_selectivity_info


# FIGURE supplementary Number of trials per neuron
num_trials_per_neuron = []
for n in metadata.keys():
    trial_ids = [e[1] for e in np.array(metadata[n])[:, sample_index]]
    num_trials_per_neuron.append(len(set(trial_ids)))
sns.set_theme(context="talk", style='white', font_scale=1.5)
fig, ax = plt.subplots(figsize=(10, 15))
# Histogram with KDE overlay
sns.histplot(num_trials_per_neuron, bins=20, kde=True, color=neuron_color, edgecolor='black', alpha=0.7, ax=ax)
ax.set_xlabel("Number of trials per neuron")
ax.set_ylabel("Count")
sns.despine()
os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)
plt.savefig(f"{save_fig_folder}/svg/num_trials_per_neuron.svg", bbox_inches='tight', format='svg')
plt.close()
plt.clf()

# FIGURE supplementary Visualize sorted average firing rate per location,
from collections import defaultdict
results = {}
for n, data in metadata.items():
    location_firing_rates = defaultdict(list)

    for sublist in data:
        location = sublist[sample_index][2]  # Extract location
        firing_rate = sublist[fr_index]  # Extract firing rate
        location_firing_rates[location].append(firing_rate)

    # Compute the average firing rate for each of the 9 locations
    avg_firing_rates = {loc: np.mean(rates) for loc, rates in location_firing_rates.items()}

    results[n] = avg_firing_rates

# add coloration based on whether the neuron is STN or nSTN
for s in range(nb_seeds):
    # Get spatial types of each neurons for this seed
    is_neuron_spatially_tuned, _, _ = get_selectivity_patterns(significant_neurons[s], minigrid_locations=False)
    spatially_tuned_neurons = [n for n in is_neuron_spatially_tuned.keys() if is_neuron_spatially_tuned[n]]
    not_spatially_tuned_neurons = [n for n in is_neuron_spatially_tuned.keys() if not is_neuron_spatially_tuned[n]]

sns.set_theme(context="talk", style='white', font_scale=1.5)
plt.figure(figsize=(10, 15))
all_sorted_rates = []  # To store sorted rates for averaging later
tuned_rates = []
not_tuned_rates = []
for neuron, rates in results.items():
    # Sort locations by firing rate (highest to lowest)
    sorted_locations = sorted(rates.keys(), key=lambda loc: rates[loc], reverse=True)
    sorted_rates = [rates[loc] for loc in sorted_locations]
    # Store for computing the average trend
    all_sorted_rates.append(sorted_rates)

    if neuron in spatially_tuned_neurons:
        color = spatially_tuned_color
        tuned_rates.append(sorted_rates)
    elif neuron in not_spatially_tuned_neurons:
        color = not_spatially_tuned_color
        not_tuned_rates.append(sorted_rates)
    else:
        color = neuron_color
    # Plot individual neuron lines in faded color
    sns.lineplot(x=range(1, len(sorted_locations) + 1), y=sorted_rates,
             color=color, linewidth=2, alpha=0.4)

# Compute and plot the average trend across all neurons
if all_sorted_rates:
    avg_trend = np.mean(tuned_rates, axis=0)  # Average across neurons
    std_trend = np.std(tuned_rates, axis=0)
    # plt.plot(range(1, len(sorted_locations) + 1), avg_trend, color=spatially_tuned_color, linewidth=6, label="STN")
    sns.lineplot(x=range(1, len(sorted_locations) + 1), y=avg_trend, #linestyle='--',
             color=spatially_tuned_color, linewidth=7, label="STN")
    plt.fill_between(range(1, len(sorted_locations) + 1), avg_trend - std_trend, avg_trend + std_trend,
                     color=spatially_tuned_color, alpha=0.3)

    avg_trend = np.mean(not_tuned_rates, axis=0)  # Average across neurons
    std_trend = np.std(not_tuned_rates, axis=0)
    # plt.plot(range(1, len(sorted_locations) + 1), avg_trend,
    #          color=not_spatially_tuned_color, linewidth=6, label="nSTN")
    sns.lineplot(x=range(1, len(sorted_locations) + 1), y=avg_trend, #linestyle='--',
             color=not_spatially_tuned_color, linewidth=7, label="nSTN")
    plt.fill_between(range(1, len(sorted_locations) + 1), avg_trend - std_trend, avg_trend + std_trend,
                     color=not_spatially_tuned_color, alpha=0.3)

plt.xlabel("Sorted spatial locations\n(Highest to lowest firing rate)")
plt.ylabel("Mean firing rate (Hz)")
plt.xticks([])
plt.legend(frameon=False)
sns.despine()
os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)
plt.savefig(f"{save_fig_folder}/svg/firing_rates_per_location.svg", bbox_inches='tight', format='svg')
plt.close()
plt.clf()


# FIGURE supplementary Visualize sorted average firing rate per Task phase
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

results = {}
for n, data in metadata.items():
    phase_firing_rates = defaultdict(list)

    for sublist in data:
        phase = np.argmax(get_trial_phases(sublist[factors_start_index:]))  # Extract phase and transform from onehot vector to integer
        firing_rate = sublist[fr_index]  # Extract firing rate
        phase_firing_rates[phase].append(firing_rate)

    # Compute the average firing rate for each of the 5 phases
    avg_firing_rates = {loc: np.mean(rates) for loc, rates in phase_firing_rates.items()}

    results[n] = avg_firing_rates

# add coloration based on whether the neuron is STN or nSTN
for s in range(nb_seeds):
    # Get spatial types of each neurons for this seed
    is_neuron_spatially_tuned, _, _ = get_selectivity_patterns(significant_neurons[s], minigrid_locations=False)
    spatially_tuned_neurons = [n for n in is_neuron_spatially_tuned.keys() if is_neuron_spatially_tuned[n]]
    not_spatially_tuned_neurons = [n for n in is_neuron_spatially_tuned.keys() if not is_neuron_spatially_tuned[n]]

sns.set_theme(context="talk", style='white', font_scale=1.5)
plt.figure(figsize=(10, 15))
all_sorted_rates = []  # To store sorted rates for averaging later
tuned_rates = []
not_tuned_rates = []
for neuron, rates in results.items():
    # Sort phases by firing rate (highest to lowest)
    sorted_phases = sorted(rates.keys(), key=lambda loc: rates[loc], reverse=True)
    sorted_rates = [rates[loc] for loc in sorted_phases]
    # Store for computing the average trend
    all_sorted_rates.append(sorted_rates)

    if neuron in spatially_tuned_neurons:
        color = spatially_tuned_color
        tuned_rates.append(sorted_rates)
    elif neuron in not_spatially_tuned_neurons:
        color = not_spatially_tuned_color
        not_tuned_rates.append(sorted_rates)
    else:
        color = neuron_color
    # Plot individual neuron lines in faded color
    sns.lineplot(x=range(1, len(sorted_phases) + 1), y=sorted_rates,
             color=color, linewidth=2, alpha=0.4)

# Compute and plot the average trend across all neurons
if all_sorted_rates:
    avg_trend = np.mean(tuned_rates, axis=0)  # Average across neurons
    std_trend = np.std(tuned_rates, axis=0)
    # plt.plot(range(1, len(sorted_phases) + 1), avg_trend, color=spatially_tuned_color, linewidth=6, label="STN")
    sns.lineplot(x=range(1, len(sorted_phases) + 1), y=avg_trend, #linestyle='--',
             color=spatially_tuned_color, linewidth=7, label="STN")
    plt.fill_between(range(1, len(sorted_phases) + 1), avg_trend - std_trend, avg_trend + std_trend,
                     color=spatially_tuned_color, alpha=0.3)

    avg_trend = np.mean(not_tuned_rates, axis=0)  # Average across neurons
    std_trend = np.std(not_tuned_rates, axis=0)

    sns.lineplot(x=range(1, len(sorted_phases) + 1), y=avg_trend, #linestyle='--',
             color=not_spatially_tuned_color, linewidth=7, label="nSTN")
    plt.fill_between(range(1, len(sorted_phases) + 1), avg_trend - std_trend, avg_trend + std_trend,
                     color=not_spatially_tuned_color, alpha=0.3)

plt.xlabel("Sorted task phases\n(Highest to lowest firing rate)")
plt.ylabel("Mean firing rate (Hz)")
plt.xticks([])
plt.legend(frameon=False)
sns.despine()

os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)
plt.savefig(f"{save_fig_folder}/svg/firing_rates_per_phase.svg", bbox_inches='tight', format='svg')
plt.close()
plt.clf()


# FIGURE supplementary Visualize firing rates
all_firing_rates = []
tuned_firing_rates = []
not_tuned_firing_rates = []

for n, data in metadata.items():
    for sublist in data:
        firing_rate = sublist[fr_index]  # Extract firing rate
        all_firing_rates.append(firing_rate)

    # Determine if the neuron is spatially tuned
    is_neuron_spatially_tuned, _, _ = get_selectivity_patterns(significant_neurons[s], minigrid_locations=False)
    if n in is_neuron_spatially_tuned and is_neuron_spatially_tuned[n]:
        tuned_firing_rates.extend([sublist[fr_index] for sublist in data])
    else:
        not_tuned_firing_rates.extend([sublist[fr_index] for sublist in data])

plt.figure(figsize=(12, 15))

# Plot histograms with KDE
bin_edges = np.histogram_bin_edges(all_firing_rates, bins=40)
# bin_edges = np.insert(bin_edges, 1, 0.5)
sns.histplot(tuned_firing_rates, bins=bin_edges, color=spatially_tuned_color, alpha=0.7, label='Spatially Tuned Neurons (STN)', kde=False)
sns.histplot(not_tuned_firing_rates, bins=bin_edges, color=not_spatially_tuned_color, alpha=0.7, label='Non-Spatially Tuned Neurons (nSTN)', kde=False)

# Set x-axis limit to 99th percentile to exclude outliers
plt.xlim([0, np.percentile(all_firing_rates, 99)])
plt.xlabel("Firing Rate (Hz)")
plt.ylabel("Number of time steps")
sns.despine()
plt.tight_layout()
os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)
plt.savefig(f"{save_fig_folder}/svg/firing_rates_distribution.svg", bbox_inches='tight', format='svg')


sns.set_theme(context="talk", style='white', font_scale=1.2)
group_names = ['Untrained ROARN w/ LR', 'ROARN w/ LR', 'Untrained ROARN', 'ROARN', 'Neurons']
bin_names = ['1,9', '2,8', '3,7', '4,6', '4,5,6', '1,2,8,9']
nb_spatial_neurons = np.zeros((len(group_names), nb_seeds))
nb_sym_spatial_neurons = np.zeros((len(group_names), nb_seeds))
nb_exl_spatial_neurons = np.zeros((len(group_names), nb_seeds))

# Compute per seeds and report the average with std (Dealt by grouped_bar_plot automatically when shapes the data correctly)
for s in range(nb_seeds):
    spatial_selectivity_neurons, sym_spatial_selectivity_neurons, exclusive_sym_spatial_selectivity_neurons = get_selectivity_patterns(significant_neurons[s], minigrid_locations=False)
    spatial_selectivity_units, sym_spatial_selectivity_units, exclusive_sym_spatial_selectivity_units = get_selectivity_patterns(significant_units[s])
    spatial_selectivity_untrained_units, sym_spatial_selectivity_untrained_units, exclusive_sym_spatial_selectivity_untrained_units = get_selectivity_patterns(significant_untrained_units[s])
    spatial_selectivity_linear_units, sym_spatial_selectivity_linear_units, exclusive_sym_spatial_selectivity_linear_units = get_selectivity_patterns(significant_linear_units[s])
    spatial_selectivity_untrained_linear_units, sym_spatial_selectivity_untrained_linear_units, exclusive_sym_spatial_selectivity_untrained_linear_units = get_selectivity_patterns(significant_untrained_linear_units[s])

    spatial_infos = [spatial_selectivity_untrained_linear_units, spatial_selectivity_linear_units, spatial_selectivity_untrained_units, spatial_selectivity_units, spatial_selectivity_neurons]
    sym_spatial_infos = [sym_spatial_selectivity_untrained_linear_units, sym_spatial_selectivity_linear_units, sym_spatial_selectivity_untrained_units, sym_spatial_selectivity_units, sym_spatial_selectivity_neurons]
    exclusive_sym_spatial_infos = [exclusive_sym_spatial_selectivity_untrained_linear_units, exclusive_sym_spatial_selectivity_linear_units, exclusive_sym_spatial_selectivity_untrained_units, exclusive_sym_spatial_selectivity_units,
                               exclusive_sym_spatial_selectivity_neurons]

    # Consider neurons with any spatial response fields, containing symmetrical fields, or with exclusively symmetrical respon fields
    for i in range(len(group_names)):
        # Count number of neurons with any spatial patterns
        count_any = 0
        sym_count_any = 0
        excl_count_any = 0
        for n in spatial_infos[i].keys():
            count_any += spatial_infos[i][n] # True if have at least one spatial response field
            sym_count_any += np.array(list(sym_spatial_infos[i][n].values())).any()
            excl_count_any += np.array(list(exclusive_sym_spatial_infos[i][n].values())).any()
        nb_spatial_neurons[i][s] = round(count_any / len(sym_spatial_infos[i].keys()) * 100)
        nb_sym_spatial_neurons[i][s] = round(sym_count_any / len(sym_spatial_infos[i].keys()) * 100)
        nb_exl_spatial_neurons[i][s] = round(excl_count_any / len(sym_spatial_infos[i].keys()) * 100)

# Merge nb_spatial_neurons with nb_excl_spatial_neurons to get data of shape (nb_group, number of seeds, number of bins)
data = np.concatenate((nb_spatial_neurons[:, :, None], nb_exl_spatial_neurons[:, :, None]), axis=2)
group_names = ['Untrained ROARN w/ LR', 'ROARN w/ LR', 'Untrained ROARN', 'ROARN', 'Neurons']
bins = ['Spatially-tuned', 'Spatially-tuned at\ntask-equivalent locations']


for i in range(len(group_names)):
    percent_spatial_neurons_is_excl = ((nb_exl_spatial_neurons[i] / nb_spatial_neurons[i])*100)
    print(f"{percent_spatial_neurons_is_excl.mean().round(1)} +- {percent_spatial_neurons_is_excl.std().round(1)} % of spatially-tuned neurons/units in {group_names[i]} are exclusively symmetrically spatially-tuned")


def plot_stacked_bar_with_individual_points(data, group_names, xlabel, ylabel, filename, folder_name=None,
                                            keep_legend=False, colors=None, rotate_xticks=True):
    if colors is None:
        colors = sns.color_palette("deep")

    # Separate data into the two components
    task_equivalent_data = data[:, :, 1]  # Data for "Spatially-tuned at task-equivalent locations" (bottom)
    spatially_tuned_data = data[:, :, 0]  # Data for "Spatially-tuned" (top)

    # Calculate means and standard deviations
    task_equivalent_mean = task_equivalent_data.mean(axis=1)
    task_equivalent_std = task_equivalent_data.std(axis=1)
    spatially_tuned_mean = spatially_tuned_data.mean(axis=1)
    spatially_tuned_std = spatially_tuned_data.std(axis=1)
    task_equivalent_perc = ((task_equivalent_mean / spatially_tuned_mean)*100).round().astype(int)

    fig, ax = plt.subplots()
    bar_width = 0.6
    num_groups = len(group_names)
    x_positions = np.arange(num_groups)  # X positions for the bars


    # Plot each bar with stacked components and individual data points
    for idx in range(num_groups):
        # Plot bottom component (Spatially-tuned at task-equivalent locations)
        ax.bar(x_positions[idx], task_equivalent_mean[idx], yerr=task_equivalent_std[idx], capsize=3, width=bar_width,
               color=colors[idx], alpha=0.9,
               hatch='//', edgecolor='black', linewidth=0)  # Add hatching to the bottom component
        # Plot top component (Spatially-tuned)
        ax.bar(x_positions[idx], spatially_tuned_mean[idx] - task_equivalent_mean[idx], yerr=spatially_tuned_std[idx], capsize=3,
               width=bar_width, bottom=task_equivalent_mean[idx],
               label=f'{group_names[idx]}',
               color=colors[idx], linewidth=0)

        # Plot individual data points for the top component
        ax.scatter([x_positions[idx]] * len(spatially_tuned_data[idx]), spatially_tuned_data[idx], color='black', s=20, alpha=0.8)
        # Plot individual data points for the bottom component
        ax.scatter([x_positions[idx]] * len(task_equivalent_data[idx]), task_equivalent_data[idx], color='black', s=20, alpha=0.8)

        if False:
            # Add percentage text in the middle of the bottom component
            ax.text(x_positions[idx], task_equivalent_mean[idx] / 2, f"{task_equivalent_perc[idx]}%", ha='center', va='center', fontsize=14, color='white')

    # Formatting
    plt.xticks([], [])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if keep_legend:
        ax.legend(frameon=False)

    sns.despine(top=True, right=True)
    if save_fig:
        if folder_name is not None:
            os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900)
            os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/svg/{filename}.svg", bbox_inches='tight', dpi=900, format='svg')
        else:
            plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', dpi=900)
            plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', format='svg')
        plt.clf()
        plt.close()
    else:
        plt.show()

#FIGURE task-equivalent %
plot_stacked_bar_with_individual_points(data, group_names, xlabel='', ylabel='Units or neurons exhibiting\nspatial tunings (%)',
                                        filename=f"nb_spatial_fields_and_sym_stacked_with_points", folder_name='spatial_receptive_field',
                                        keep_legend=True, colors=[untrained_linear_color, linear_color, untrained_color, roarn_color, neuron_color])



############################################
# Histogram number of spatial response fields (untrained vs trained vs neurons)
#  also distance / similarity metric between untrained and neurons distribution and between trained and neurons distribution
############################################
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
                    linewidth=linewidth, linestyle=linestyle if i % 2 == 0 else '-') # linewidth=2

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
        os.makedirs(f"{save_fig_folder}/spatial_receptive_field/", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/spatial_receptive_field/{filename}", bbox_inches='tight', dpi=900)
        os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/{filename}.svg", bbox_inches='tight', dpi=900, format='svg')
        plt.clf()
    else:
        plt.show()

# Plot grouped bar plot
group_names = ['Untrained ROARN w/ LR', 'ROARN w/ LR', 'Untrained ROARN', 'ROARN', 'Neurons']
significant_unit_types = [significant_untrained_linear_units, significant_linear_units,
                          significant_untrained_units, significant_units, significant_neurons]
bin_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
count_per_type = np.zeros((len(group_names), nb_seeds, len(bin_names)))
# nb_fields_distribution shape = (nb of models, nb_seeds, number of neurons/units)
nb_fields_distribution = [[[] for _ in range(nb_seeds)] for _ in range(len(group_names))]
for s in range(nb_seeds):
    for i in range(len(group_names)):
        nb_significant_locations = []
        for n in significant_unit_types[i][s].keys():
            nb_significant_locations.append(np.array(list(significant_unit_types[i][s][n].values())).sum())

        nb_fields_distribution[i][s] = nb_significant_locations
        count = Counter(nb_significant_locations)
        for j in count.keys():
            count_per_type[i][s][j] = round(count[j] / sum(count.values()) * 100)

# Can't average across seeds, because don't expect trained units 123 to have the same number of spatial fields across seeds
# plot_distribution_hist([np.mean(nb_fields_distribution[i], axis=0) for i in range(len(group_names))], 'distribution_number_spatial_response_field_averaged_across_seeds')
sns.set_theme(context="talk", style='white', font_scale=1.2)
data = [np.concatenate(nb_fields_distribution[i]) for i in range(len(group_names))]

#FIGURE response fields
plot_distribution_hist(data[-3:], x_label="Number of spatial response fields",
                       y_label="Proportion of the population",
                       legend_labels=None,  # legend_labels=group_names[::-1],
                       filename='distribution_number_spatial_response_field_all_seeds_concatenated',
                       colors=[untrained_color, roarn_color, neuron_color], add_mean=False)

#FIGURE response fields LR
plot_distribution_hist(data[:2] + [data[-1]], x_label="Number of spatial response fields",
                       y_label="Proportion of the population",
                       legend_labels=None,  # legend_labels=group_names[::-1],
                       filename='distribution_number_spatial_response_field_all_seeds_concatenated_LR',
                       colors=[untrained_linear_color, linear_color, neuron_color], add_mean=False)

plot_distribution_hist(data, x_label="Number of spatial response fields",
                       y_label="Proportion of the population",
                       legend_labels=group_names,  # legend_labels=group_names[::-1],
                       filename='distribution_number_spatial_response_field_all_seeds_concatenated_all_models',
                       colors=[untrained_linear_color, linear_color,untrained_color, roarn_color, neuron_color], add_mean=False)

perc_non_spatial_units = (np.array(nb_fields_distribution[1]) == 0).sum(axis=1)  / len(nb_fields_distribution[1][0]) * 100
print(f"{perc_non_spatial_units.mean().round(1)} +- {perc_non_spatial_units.std().round(1)} % of trained units are not spatially-tuned")

perc_spatial_neurons = (np.array(nb_fields_distribution[2]) > 0).sum(axis=1)  / len(nb_fields_distribution[2][0]) * 100
print(f"{perc_spatial_neurons.mean().round(1)} +- {perc_spatial_neurons.std().round(1)} % of neurons are spatially-tuned")

avg_nb_fields = [np.array(nb_fields_distribution[i]).mean(axis=1) for i in range(len(group_names))]
print(f"Average number of spatial response fields per units/neurons for untrained, trained, neurons: {np.array(avg_nb_fields).mean(axis=1).round()} +- {np.array(avg_nb_fields).std(axis=1).round()}")



#FIGURE Neurons fields distribution
im = sns.kdeplot(data[-1], fill=True, cut=0, color=darker_neuron_color, alpha=0.4, linewidth=2, linestyle='-')
im = sns.histplot(data[-1], stat='density', kde=True, discrete=True, shrink=0.7, color=darker_neuron_color, alpha=0.8, edgecolor='gray', linewidth=0.5) # shrink=0.5, color=neuron_color, edgecolor=None, linewidth=0)
plt.ylabel("Density")
plt.xlabel("Number of Spatial Response Fields")
plt.xlim(0, np.max(data[-1]))  # Adding padding to x-axis limit
plt.yticks([0.1, 0.3, 0.5])  # Fine-tuned tick marks
sns.despine(top=True, right=True)

# Save or display plot
if save_fig:
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/distribution_number_spatial_response_field_neurons_only", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/distribution_number_spatial_response_field_neurons_only.svg", bbox_inches='tight', dpi=900,
                format='svg')
    plt.clf()
    plt.close()
else:
    plt.show()


#FIGURE Pie spatially vs non-spatially tuned
spatial_neurons_count = np.zeros((nb_seeds,))
spatial_neurons_count_excl = np.zeros((nb_seeds,))
for s in range(nb_seeds):
    spatial_selectivity_neurons, sym_spatial_selectivity_neurons, exclusive_sym_spatial_selectivity_neurons = get_selectivity_patterns(significant_neurons[s], minigrid_locations=False)
    # Consider neurons with any spatial response ields, containing symmetrical fields, or with exclusively symmetrical respon fields
    # Count number of neurons with any spatial patterns
    count_any = 0
    excl_count_any = 0
    for n in spatial_selectivity_neurons.keys():
        count_any += spatial_selectivity_neurons[n] # True if have at least one spatial response field
        excl_count_any += np.array(list(exclusive_sym_spatial_selectivity_neurons[n].values())).any()
    spatial_neurons_count[s] = count_any
    spatial_neurons_count_excl[s] = excl_count_any

perc_spatial_neurons = ((np.array(nb_fields_distribution[-1]) > 0).sum(axis=1) / len(nb_fields_distribution[-1][0]) * 100).mean()
perc_non_spatial_neurons = ((np.array(nb_fields_distribution[-1]) == 0).sum(axis=1) / len(nb_fields_distribution[-1][0]) * 100).mean()
perc_spatial_neurons_task_specific_among_spatial_neurons = ((nb_exl_spatial_neurons[-1] / nb_spatial_neurons[-1])*100)
perc_spatial_neurons_task_specific = (spatial_neurons_count_excl / len(nb_fields_distribution[-1][0]) * 100).mean().round()
perc_spatial_neurons = (spatial_neurons_count / len(nb_fields_distribution[-1][0]) * 100).mean()
data = [perc_non_spatial_neurons, perc_spatial_neurons, perc_spatial_neurons_task_specific]

sns.set_theme(context="talk", style='white', font_scale=1.4)
fig, ax = plt.subplots()
size = 0.1
inner_colors = [not_spatially_tuned_color, spatially_tuned_color]
outer_colors = ['#00000000', '#badaff']
labels_inner =  [f"Non-spatially tuned\n{int(data[0].round())}%", f'Spatially tuned\n{int(data[1].round())}%']
labels_outer = ['', f'Spatially tuned\nat task-equivalent locations\n{int(data[-1].round())}%']
# outer slices
outer_thickness = 0.07 #0.05
ax.pie([data[0], data[-1]], radius=1 - size + outer_thickness, colors=outer_colors,
       wedgeprops=dict(width=outer_thickness, edgecolor='w'), labels=labels_outer) # , autopct='%1.0f%%', pctdistance=1.25
# inner slices
ax.pie(data[0:2], radius=1 - size, colors=inner_colors,
       wedgeprops=dict(width=1 - size - 0.2, edgecolor='w', linewidth=0), labels=labels_inner,
       labeldistance=0.6)  # ,  autopct='%1.0f%%', pctdistance=1.25

fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
ax.axis('equal')
ax.margins(0, 0)

if save_fig:
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/pie_spatial_tuning_neurons_only", bbox_inches='tight')
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/pie_spatial_tuning_neurons_only.svg", bbox_inches='tight',
                format='svg')
    plt.clf()
else:
    plt.tight_layout()
    plt.show()


from scipy.stats import wasserstein_distance
def plot_wasserstein_distance(data, model_names, nb_seeds, colors=None, keep_xticks=True,
                              filename='distribution_wasserstein_distance', foldername='spatial_receptive_field',
                              title=None, comparison_labels=None, yticks=None,
                              baseline_value = None, baseline_label = None):
    """
    data: shape = (nb of models, nb_seeds, number of samples)
    if comparison_labels is not None:
    data: list [(2 i.e. the two distributions to compute the distance, nb_seeds, number of samples) * nb of comparison_labels]
    """
    if colors is None:
        colors = sns.set_palette(sns.color_palette("deep"))
    if comparison_labels is None:
        neurons_index = model_names.index('Neurons')
        xlabels = [e for e in model_names if e != 'Neurons']
        x_positions = np.arange(len(model_names) - 1)
        distances = [[] for _ in range(len(model_names) - 1)]
        for i in range(len(model_names)):
            if i == neurons_index:
                continue
            for s in range(nb_seeds):
                # without_nan = (np.isnan(data[i][s]) + np.isnan(data[neurons_index][s])) == False
                distances[i].append(wasserstein_distance(data[i][s], data[neurons_index][s]))
    else:
        xlabels = comparison_labels
        x_positions = np.arange(len(comparison_labels))
        distances = [[] for _ in range(len(comparison_labels))]
        for s in range(nb_seeds):
            for dist, data_subset in zip(distances, data):
                without_nan = (np.isnan(data_subset[0][s]) + np.isnan(data_subset[1][s])) == False
                dist.append(wasserstein_distance(data_subset[0][s][without_nan], data_subset[1][s][without_nan]))
    ax = sns.barplot(x=xlabels, y=np.mean(distances, axis=1), palette=colors)
    for i in range(len(x_positions)):
        ax.scatter([x_positions[i]] * len(distances[i]), distances[i], color='black', s=20, alpha=0.8)
    # ax.bar_label(ax.containers[0], fmt='%.2f', padding=3)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    plt.errorbar(x=x_coords, y=y_coords, yerr=np.std(distances, axis=1), fmt="none", c="k", capsize=3)  # , elinewidth=0.5
    if title is not None:
        plt.title(title)

    if keep_xticks:
        plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
    else:
        plt.xticks([], [])
    if yticks is not None:
        plt.yticks(yticks)
    plt.tight_layout()
    sns.despine(top=True, right=True)

    # Adding baseline values as dashed horizontal lines
    if baseline_value is not None and baseline_label is not None:
        lines = []
        for i, (pos, value) in enumerate(zip(x_positions, baseline_value)):
            line = plt.axhline(y=value, xmin=(pos+0.15)/len(x_positions), xmax=(pos+0.85)/len(x_positions), linestyle='dashed', color='black')
            lines.append(line)
        plt.legend(handles=lines, labels=[baseline_label], frameon=False)

    if save_fig:
        if foldername is not None:
            os.makedirs(f"{save_fig_folder}/{foldername}/", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/{foldername}/{filename}", bbox_inches='tight', dpi=900)
            os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/svg/{filename}.svg", bbox_inches='tight', format='svg')
        else:
            plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight')
            plt.savefig(f"{save_fig_folder}/{filename}.svg", bbox_inches='tight', format='svg')
        plt.clf()
        plt.close()
    else:
        plt.show()

    if comparison_labels is not None:
        for i, j in combinations(range(len(comparison_labels)), 2):
            res = stats.ranksums(np.array(distances)[i], np.array(distances)[j], alternative='less', nan_policy='omit', axis=None)
            print(
                f"Distance between {model_names[0]} and {model_names[1]} increases from {comparison_labels[i]} to {comparison_labels[j]}: "
                f"d-statistic={res.statistic} pvalue={res.pvalue} Wilcoxon rank-sum test")

        if model_names[0] == 'Ideal observer' and model_names[1] == 'ROARN w/ LR':
            res = stats.ranksums(np.array(distances)[0], np.array(distances)[1],
                                 alternative='greater', nan_policy='omit', axis=None)
            print(
                f"Distance between {model_names[0]} and {model_names[1]} decreases from {comparison_labels[0]} to {comparison_labels[1]}: d-statistic={res.statistic} pvalue={res.pvalue} Wilcoxon rank-sum test")


sns.set_theme(context="talk", style='white', font_scale=1.5)
#FIGURE
plot_wasserstein_distance(nb_fields_distribution, group_names, nb_seeds,
                          colors=[untrained_linear_color, linear_color, untrained_color, roarn_color, neuron_color],
                          keep_xticks=False, yticks=[0, 1.5, 2.5])

#FIGURE
# Bar plot similarity betweeen histogram
sns.set_theme(context="talk", style='white', font_scale=1.5)
neurons_index = group_names.index('Neurons')
untrained_data = np.min(count_per_type[[group_names.index('Untrained ROARN'), neurons_index]], axis=0).sum(axis=-1)
trained_data = np.min(count_per_type[[group_names.index('ROARN'), neurons_index]], axis=0).sum(axis=-1)
untrained_linear_data = np.min(count_per_type[[group_names.index('Untrained ROARN w/ LR'), neurons_index]], axis=0).sum(axis=-1)
trained_linear_data = np.min(count_per_type[[group_names.index('ROARN w/ LR'), neurons_index]], axis=0).sum(axis=-1)
data = [untrained_linear_data, trained_linear_data, untrained_data, trained_data]
ax = sns.barplot(x=[e for e in group_names if e != 'Neurons'], y=np.mean(data,axis=1),
                 palette=[untrained_linear_color, linear_color, untrained_color, roarn_color], linewidth=0)
x_positions = np.arange(len(data))
for i in range(len(x_positions)):
    ax.scatter([x_positions[i]] * len(data[i]), data[i], color='black', s=20, alpha=0.8)
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=[np.std(untrained_linear_data), np.std(trained_linear_data),
                                           np.std(untrained_data), np.std(trained_data)], fmt="none", c="k", capsize=3)
plt.ylabel("Distribution overlap (%)")
plt.tight_layout()
if False:
    plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
else:
    plt.xticks([], [])
sns.despine(top=True, right=True)
if save_fig:
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/distribution_intersection", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/distribution_intersection.svg",
        bbox_inches='tight', format='svg')
    plt.clf()
sns.reset_orig()



############################################
# Grouped bar plot where
#   y-axis = untrained normalized neural predictivity or Neural predictiviy gain or Neural predictivity
#   bin = ['Spatially-tuned', 'Not-spatially tuned]
#   group = ['Untrained', 'Location', 'Ideal observer', 'EPN', 'ROARN']
############################################
def get_regression_dict(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]

    result_per_neuron = {}
    for line in lines:
        regressor = line[line.find("Regressor: ") + len("Regressor: "):line.find(", Best_result:")]
        if regressor == 'LinearRegression' or regressor == 'Ridge':
            continue
        neuron_id = line[8:line.find(", Regressor: ")]
        runtime = float(line[line.find("Runtime: ") + len("Runtime: "):line.find(" seconds")])
        try:
            best_result = float(line[line.find("Best_result:") + len("Best_result: "):line.find(", Param:")])
            old_file_version = True
        except ValueError:
            old_file_version = False
        if old_file_version:
            r2 = float(line[line.find("R2: ") + len("R2: "):line.find(", MSE:")])
            mse = float(line[line.find("MSE: ") + len("MSE: "):line.find(", Pearsonr:")])
            if 'Spearmanr' in line:
                corr = float(line[line.find("Pearsonr: ") + len("Pearsonr: "):line.find(", Spearmanr:")])
                rank_corr = float(line[line.find("Spearmanr: ") + len("Spearmanr: "):])
            else:
                corr = float(line[line.find("Pearsonr: ") + len("Pearsonr: "):])
            if "Train_window" in line:
                train_window = int(
                    line[line.find("Train_window: ") + len("Train_window: "):line.find(", Test_window:")])
                test_window = int(line[line.find("Test_window: ") + len("Test_window: "):line.find(", R2:")])
        else:
            if 'Spearmanr' in line:
                corr = float(line[line.find("Pearsonr: ") + len("Pearsonr: "):line.find(", Spearmanr:")])
            else:
                corr = float(line[line.find("Pearsonr: ") + len("Pearsonr: "):])
        if not neuron_id in result_per_neuron.keys():
            # result_per_neuron[neuron_id]['Best_result'] = [(regressor, best_result)]
            if "Time_window" in line:
                result_per_neuron[neuron_id] = {'Best_result': [(regressor, best_result)],
                                                'Runtime': [(regressor, runtime)],
                                                'R2': [(regressor, r2)],
                                                'MSE': [(regressor, mse)],
                                                'Corr': [(regressor, corr)],
                                                'Rank_corr': [(regressor, rank_corr)],
                                                'Train_window': [(regressor, train_window)],
                                                'Test_window': [(regressor, test_window)],
                                                }
            elif old_file_version and 'Spearmanr' in line:
                result_per_neuron[neuron_id] = {'Best_result': [(regressor, best_result)],
                                                'Runtime': [(regressor, runtime)],
                                                'R2': [(regressor, r2)],
                                                'MSE': [(regressor, mse)],
                                                'Corr': [(regressor, corr)],
                                                'Rank_corr': [(regressor, rank_corr)],
                                                }
            elif old_file_version:
                result_per_neuron[neuron_id] = {'Best_result': [(regressor, best_result)],
                                                'Runtime': [(regressor, runtime)],
                                                'R2': [(regressor, r2)],
                                                'MSE': [(regressor, mse)],
                                                'Corr': [(regressor, corr)],
                                                }
            else:
                result_per_neuron[neuron_id] = {'Runtime': [(regressor, runtime)],
                                                'Corr': [(regressor, corr)],
                                                }

        else:
            result_per_neuron[neuron_id]['Runtime'].append((regressor, runtime))
            result_per_neuron[neuron_id]['Corr'].append((regressor, corr))
            if old_file_version:
                result_per_neuron[neuron_id]['Best_result'].append((regressor, best_result))
                result_per_neuron[neuron_id]['R2'].append((regressor, r2))
                result_per_neuron[neuron_id]['MSE'].append((regressor, mse))
            if old_file_version and 'Spearmanr' in line:
                result_per_neuron[neuron_id]['Rank_corr'].append((regressor, rank_corr))
            if "Train_window" in line:
                result_per_neuron[neuron_id]['Train_window'].append((regressor, train_window))
                result_per_neuron[neuron_id]['Test_window'].append((regressor, test_window))

    return result_per_neuron

def get_regression_model_specific_performance(reg_type, filenames,repeated_seeds=None,subselection_neurons=None, metric='Corr'):
    scores = []
    for filename in filenames:
        result_per_neuron = get_regression_dict(filename)
        model_score = []
        if subselection_neurons is None:
            subselection_neurons = result_per_neuron.keys()
        for neuron in subselection_neurons:
            # Select the correlation score obtained with regression model 'reg_type'
            model_score.append([e[1] for e in result_per_neuron[neuron][metric] if e[0] == reg_type][0])
        scores.append(model_score)
    return scores

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
            "reg_output/memory-cc-v2_linear_rnn_3_4-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_linear_rnn_3_5-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_linear_rnn_3_6-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_rnn_3_4-k10/344.txt",
            "reg_output/memory-cc-v2_rnn_3_5-k10/343.txt",
            "reg_output/memory-cc-v2_rnn_3_6-k10/383.txt",
        ]

names = ['Untrained ROARN', 'Position', 'Ideal observer', 'ROARN w/ LR', 'ROARN']
repeated_seeds = [3, 3, 3, 3, 3]

# Shape: (nb_seeds, number of models i.e. len(names), number of neurons that were classified as this spatial type for this seed)
spatially_tuned_scores = []
not_spatially_tuned_scores = []
all_neurons = []
for s in range(nb_seeds):
    # Get spatial types of each neurons for this seed
    is_neuron_spatially_tuned, _, _ = get_selectivity_patterns(significant_neurons[s], minigrid_locations=False)
    spatially_tuned_neurons = [n for n in is_neuron_spatially_tuned.keys() if is_neuron_spatially_tuned[n]]
    not_spatially_tuned_neurons = [n for n in is_neuron_spatially_tuned.keys() if not is_neuron_spatially_tuned[n]]

    # Select files of the 's'-th seed of each model
    subselection_files = np.array(filenames)[np.arange(len(filenames), step=3)+s]
    # Get neural predictivity for each models and accumulate them for each seeds of neuron spatial types classification
    spatially_tuned_scores.append(get_regression_model_specific_performance('LinearSVR', subselection_files, None,
                                                               spatially_tuned_neurons))
    not_spatially_tuned_scores.append(get_regression_model_specific_performance('LinearSVR', subselection_files, None,
                                                                  not_spatially_tuned_neurons))

    all_neurons.append(get_regression_model_specific_performance('LinearSVR', subselection_files, None, None))


#  1. Get mean and median np for each (model, seed)
#  2. Shape data as: (2 (spatially tuned or not), nb_seeds, nb of model)
#  3. Adjust the avg np of each model by the untrained np of that seed
#  4. Give data to grouped_bar_plot

#FIGURE
sns.set_theme(context="talk", style='white', font_scale=1)
# Shape: (nb_seeds, number of models i.e. len(names), number of neurons that were classified as this spatial type for this seed)
mean_spatially_tuned_scores = np.zeros((nb_seeds, len(names)))
mean_not_spatially_tuned_scores = np.zeros((nb_seeds, len(names)))
mean_all_neuron_scores = np.zeros((nb_seeds, len(names)))
for s in range(nb_seeds):
    for model in range(len(names)):
        mean_spatially_tuned_scores[s, model] = np.nanmean(spatially_tuned_scores[s][model])
        mean_not_spatially_tuned_scores[s, model] = np.nanmean(not_spatially_tuned_scores[s][model])
        mean_all_neuron_scores[s, model] = np.nanmean(all_neurons[s][model])
mean_data = np.stack((mean_spatially_tuned_scores, mean_not_spatially_tuned_scores))
untrained_normed_mean_data = (mean_data / mean_data[:, :, 0][:, :, None])

sns.set_theme(context="talk", style='white', font_scale=1.6)
remove_linear_roarn = False
if remove_linear_roarn:
    filter = np.delete(np.arange(len(names)), names.index('ROARN w/ LR'))
    mean_all_neuron_scores = mean_all_neuron_scores[:,filter]
    ax = sns.barplot(x=np.array(names)[filter], y=mean_all_neuron_scores.mean(axis=0),
                     palette=np.array(model_colors,dtype=object)[filter])
else:
    ax = sns.barplot(x=names, y=mean_all_neuron_scores.mean(axis=0), palette=model_colors)
for i in range(mean_all_neuron_scores.shape[1]):
    ax.scatter([i] * nb_seeds, mean_all_neuron_scores[:,i], color='black', s=20, alpha=0.8)
ax.bar_label(ax.containers[0], fmt='%.2f', padding=16)
ax.axes.get_yaxis().set_ticks([])
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=mean_all_neuron_scores.std(axis=0), fmt="none", c="k", capsize=3)  # , elinewidth=0.5
plt.ylabel("Neural predictivity")
sns.despine(top=True, right=True)
if False:
    plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
else:
    plt.xticks([], [])
plt.tight_layout()
if save_fig:
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/np_all_neurons",
                bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/np_all_neurons.svg", bbox_inches='tight', dpi=900, format='svg')
    plt.clf()
    plt.close()
else:
    plt.show()


#FIGURE
bins = ['Spatially tuned', 'Non-spatially tuned']
filter = np.delete(np.arange(len(names)), names.index('ROARN w/ LR'))
untrained_substracted_mean_data = (mean_data - mean_data[:, :, 0][:, :, None])[:,:,filter]
sns.set_theme(context="talk", style='white', font_scale=1.2)
grouped_bar_plot(np.transpose(untrained_substracted_mean_data, (2,1,0)), np.array(names)[filter],
                 bins, error_type='std', xlabel='',
                 ylabel='Neural predictivity gain',
                 filename=f"np_spatial_vs_not_untrained_normalized_substracted", folder_name='spatial_receptive_field',
                 keep_legend=False, rotate_xticks=False, colors=np.array(model_colors,dtype=object)[filter],
                 add_scatterpoints=True)

# Test if gap betweeen IO and ROARN is significantly bigger for no-spatially tuned neurons
untrained_spatially_tuned_np = [np.array(spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
untrained_non_spatially_tuned_np = [np.array(not_spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
spatially_tuned_np_gain = []
non_spatially_tuned_np_gain = []
subselection_spatially_tuned_scores = [] # shape: (number of models that was subselected, nb_seeds, number of samples)
subselection_not_spatially_tuned_scores = []

names_of_interest =  ['Ideal observer', 'ROARN']
for m in names_of_interest:
    subselection_spatially_tuned_scores.append([np.array(spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_spatially_tuned_np_gain = [np.array(spatially_tuned_scores[s][names.index(m)]) - untrained_spatially_tuned_np[s] for s in range(nb_seeds)]
    spatially_tuned_np_gain.append(model_spatially_tuned_np_gain)

    subselection_not_spatially_tuned_scores.append([np.array(not_spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_non_spatially_tuned_np_gain = [np.array(not_spatially_tuned_scores[s][names.index(m)]) - untrained_non_spatially_tuned_np[s] for s in range(nb_seeds)]
    non_spatially_tuned_np_gain.append(model_non_spatially_tuned_np_gain)

# NP gain gap
gap_between_io_and_rnn_spatially_tuned = [spatially_tuned_np_gain[1][s] - spatially_tuned_np_gain[0][s] for s in range(nb_seeds)]
gap_between_io_and_rnn_nonspatially_tuned = [non_spatially_tuned_np_gain[1][s] - non_spatially_tuned_np_gain[0][s] for s in range(nb_seeds)]

data_flat = np.concatenate([np.concatenate(gap_between_io_and_rnn_spatially_tuned), np.concatenate(gap_between_io_and_rnn_nonspatially_tuned)])
df = pd.DataFrame.from_dict({
    f'Gap between neural predictivity\ngain of {names_of_interest[1]} and {names_of_interest[0]}': data_flat,
    'tuning': np.concatenate([
        ['Spatially-tuned' for _ in range(np.concatenate(gap_between_io_and_rnn_spatially_tuned).size)],
        ['Non-spatially tuned' for _ in range(np.concatenate(gap_between_io_and_rnn_nonspatially_tuned).size)]
    ]),
    '': ['' for _ in range(data_flat.size)]
})


#FIGURE maybe if want kernel density estimate CDF instead of empirical cumulative distribution function
f = sns.kdeplot(data=df, x=f"Gap between neural predictivity\ngain of {names_of_interest[1]} and {names_of_interest[0]}", hue="tuning", multiple="layer", cumulative=True)
sns.despine(top=True, right=True)
f.get_legend().set_title(None)
f.get_legend().get_frame().set_linewidth(0.0)
if save_fig:
    plt.savefig(f"{save_fig_folder}/cumulative_distribution_gap_np_gain_rnn_io_KDE", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/cumulative_distribution_gap_np_gain_rnn_io_KDE.svg", bbox_inches='tight', dpi=900,
                format='svg')
    plt.clf()
    plt.close()
else:
    plt.show()

#FIGURE
# Empirical cumulative distribution function
sns.displot(data=df, x=f"Gap between neural predictivity\ngain of {names_of_interest[1]} and {names_of_interest[0]}",
            hue="tuning", kind="ecdf")
plt.ylabel("Cumulative proportion")
if save_fig:
    plt.savefig(f"{save_fig_folder}/cumulative_distribution_gap_np_gain_rnn_io", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/cumulative_distribution_gap_np_gain_rnn_io.svg", bbox_inches='tight', dpi=900,
                format='svg')
    plt.clf()
    plt.close()
else:
    plt.show()


############################################
# Compared neural predictivity gain between IO and ROARN
# Scatter plot and histogram
############################################
def plot_scatter(title, x, y, x2=None, y2=None, x3=None, y3=None,
                 x_label='', y_label='',legend=None, save_fig=True, add_correlation=False, filename='', keep_legend=True,
                 foldername=None, colors=None):
    if colors is None:
        colors = sns.color_palette('deep')
    plt.scatter(x, y, alpha=0.9, c=[colors[0]])
    max_val = np.nanmax([np.nanmax(x), np.nanmax(y)])
    min_val = np.nanmin([np.nanmin(x), np.nanmin(y)])
    if x2 is not None:
        plt.scatter(x2, y2, alpha=0.8, c=[colors[1]])
        if legend is not None:
            plt.legend(legend)
        max_val = np.nanmax([max_val, np.nanmax(x2), np.nanmax(y2)])
        min_val = np.nanmin([min_val, np.nanmin(x2), np.nanmin(y2)])
        if x3 is not None:
            plt.scatter(x3, y3, alpha=0.8, c=[colors[2]])
            if legend is not None:
                plt.legend(legend)
            max_val = np.nanmax([max_val, np.nanmax(x3), np.nanmax(y3)])
            min_val = np.nanmin([min_val, np.nanmin(x3), np.nanmin(y3)])
    plt.title(f"{title}")
    plt.plot([min_val, max_val], [min_val, max_val], '--', alpha=0.5, color='black')
    if add_correlation:
        without_nan = (np.isnan(x) + np.isnan(y)) == False
        corr, pvalue = pearsonr(np.array(x)[without_nan], np.array(y)[without_nan])
        plt.annotate("$\it{r}$" + f" = {corr:.2f}", (0.01, 0.94), xycoords='axes fraction')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.xticks(np.arange(-0.1, max_val.round(1)+0.1, 0.2))
    plt.yticks(np.arange(-0.1, max_val.round(1)+0.1, 0.2))

    ax = plt.gca()
    if not keep_legend:
        sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, -0.35), ncol=4, title=None, frameon=False)
        ax.get_legend().remove()
    else:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # Put a legend to the right of the current axis
        ax.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine(top=True, right=True)
    if save_fig:
        if foldername is not None:
            os.makedirs(f"{save_fig_folder}/{foldername}/", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/{foldername}/scatter-{filename}", bbox_inches='tight', dpi=900)
            os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/svg/scatter-{filename}.svg", bbox_inches='tight', format='svg')
        else:
            plt.savefig(f"{save_fig_folder}/scatter-{filename}", bbox_inches='tight')
            plt.savefig(f"{save_fig_folder}/scatter-{filename}.svg", bbox_inches='tight', format='svg')
        plt.clf()
        plt.close()
    else:
        plt.show()


untrained_spatially_tuned_np = [np.array(spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
untrained_non_spatially_tuned_np = [np.array(not_spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
spatially_tuned_np_gain = []
non_spatially_tuned_np_gain = []
subselection_spatially_tuned_scores = [] # shape: (number of models that was subselected, nb_seeds, number of samples)
subselection_not_spatially_tuned_scores = []
names_of_interest =  ['Ideal observer', 'ROARN']
colors_of_interest = [io_color, roarn_color]
for m in names_of_interest:
    subselection_spatially_tuned_scores.append([np.array(spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_spatially_tuned_np_gain = [np.array(spatially_tuned_scores[s][names.index(m)]) - untrained_spatially_tuned_np[s] for s in range(nb_seeds)]
    spatially_tuned_np_gain.append(model_spatially_tuned_np_gain)

    subselection_not_spatially_tuned_scores .append([np.array(not_spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_non_spatially_tuned_np_gain = [np.array(not_spatially_tuned_scores[s][names.index(m)]) - untrained_non_spatially_tuned_np[s] for s in range(nb_seeds)]
    non_spatially_tuned_np_gain.append(model_non_spatially_tuned_np_gain)
spatially_tuned_np_gain_flat = [np.concatenate(spatially_tuned_np_gain[i]) for i in range(len(spatially_tuned_np_gain))]
non_spatially_tuned_np_gain_flat = [np.concatenate(non_spatially_tuned_np_gain[i]) for i in range(len(non_spatially_tuned_np_gain))]
#FIGURE or maybe more appendix
plot_scatter(title='', x=spatially_tuned_np_gain_flat[1], y=spatially_tuned_np_gain_flat[0],
             x2=non_spatially_tuned_np_gain_flat[1], y2=non_spatially_tuned_np_gain_flat[0],
             x_label=f'{names_of_interest[1]} neural predictivity gain', y_label=f'{names_of_interest[0]}\nneural predictivity gain',
             legend=['Spatially-tuned', 'Non-spatially tuned'], save_fig=True, filename='spatially_vs_not_np_gain_io_rnn',
             keep_legend=False)


# Correaltion of raw np across all seeds
x, y, x2, y2 = spatially_tuned_np_gain_flat[1], spatially_tuned_np_gain_flat[0], non_spatially_tuned_np_gain_flat[1], non_spatially_tuned_np_gain_flat[0]
without_nan = (np.isnan(x) + np.isnan(y)) == False
corr, pvalue = pearsonr(np.array(x)[without_nan], np.array(y)[without_nan])
without_nan = (np.isnan(x2) + np.isnan(y2)) == False
corr2, pvalue2 = pearsonr(np.array(x2)[without_nan], np.array(y2)[without_nan])
print(f"Neural predictivity of IO and ROARN is more correlated across spatially-tuned neurons "
      f"{corr} than non-spatially tuned {corr2}")

# Correlation of np gain per seeds
spatially_tuned_corrs = []
spatially_tuned_pvalues = []
non_spatially_tuned_corrs = []
non_spatially_tuned_pvalues = []
for s in range(nb_seeds):
    # Correlation between the np gain of Ideal observer and ROARN for each seeds
    # Only np gain for spatially tuned neurons
    without_nan = (np.isnan(spatially_tuned_np_gain[0][s]) + np.isnan(spatially_tuned_np_gain[1][s])) == False
    spatially_tuned_corr, spatially_tuned_pvalue = pearsonr(spatially_tuned_np_gain[0][s][without_nan], spatially_tuned_np_gain[1][s][without_nan])

    without_nan = (np.isnan(non_spatially_tuned_np_gain[0][s]) + np.isnan(non_spatially_tuned_np_gain[1][s])) == False
    non_spatially_tuned_corr, non_spatially_tuned_pvalue = pearsonr(non_spatially_tuned_np_gain[0][s][without_nan],
                                                                    non_spatially_tuned_np_gain[1][s][without_nan])
    spatially_tuned_corrs.append(spatially_tuned_corr)
    spatially_tuned_pvalues.append(spatially_tuned_pvalue)
    non_spatially_tuned_corrs.append(non_spatially_tuned_corr)
    non_spatially_tuned_pvalues.append(non_spatially_tuned_pvalue)
print(f"Neural predictivity of {names_of_interest[0]} and {names_of_interest[1]} is more correlated across spatially-tuned neurons gain"
      f"{np.mean(spatially_tuned_corrs).round(3)} +-{np.std(spatially_tuned_corrs).round(3)} p={spatially_tuned_pvalues} than non-spatially"
      f" tuned {np.mean(non_spatially_tuned_corrs).round(3)} +-{np.std(non_spatially_tuned_corrs).round(3)} p={non_spatially_tuned_pvalues}")


#FIGURE correlation np gain spatial vs not
neuron_tuning = ['Spatially-tuned', 'Non-spatially tuned']
ax = sns.barplot(x=neuron_tuning, y=[np.mean(spatially_tuned_corrs), np.mean(non_spatially_tuned_corrs)],
                 palette=[spatially_tuned_color, not_spatially_tuned_color]) # sns.set_palette(sns.color_palette("deep"))
ax.bar_label(ax.containers[0], fmt='%.2f', padding=6)
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=[np.std(spatially_tuned_corrs), np.std(non_spatially_tuned_corrs)], fmt="none", c="k", capsize=3)  # , elinewidth=0.5
sns.set_theme(context="talk", style='white', font_scale=1)
plt.ylabel(r"Pearson $\rho$")
x_positions = np.arange(len(neuron_tuning))
data = [spatially_tuned_corrs, non_spatially_tuned_corrs]
for i in range(len(x_positions)):
    ax.scatter([x_positions[i]] * len(data[i]), data[i], color='black', s=20, alpha=0.8)
plt.yticks([], [])
plt.xticks([], [])
plt.tight_layout()
box = ax.get_position() # Make plot thinner
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
sns.despine(top=True, right=True)
if save_fig:
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/correlation_np_gain_io_rnn_spatially_vs_not",
                bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/correlation_np_gain_io_rnn_spatially_vs_not.svg", bbox_inches='tight', format='svg')
    plt.clf()
    plt.close()


#FIGURE maybe in appendix
sns.set_theme(context="talk", style='white', font_scale=1.2)
plot_distribution_hist(spatially_tuned_np_gain_flat, x_label='Neural predictivity gain', y_label='Proportion of neurons (%)',
                       legend_labels=[f'{names_of_interest[0]}', f'{names_of_interest[1]}'], filename='hist_spatially_tuned',
                       discret_hist=False, bins=8, colors=colors_of_interest, xticks=np.arange(-.2, 1, 0.2))
plot_distribution_hist(non_spatially_tuned_np_gain_flat, x_label='Neural predictivity gain', y_label='Proportion of neurons (%)',
                       legend_labels=[f'{names_of_interest[0]}', f'{names_of_interest[1]}'], filename='hist_non_spatially_tuned',
                       discret_hist=False, bins=8, colors=colors_of_interest, xticks=np.arange(-.2, 1, 0.2))


#FIGURE
# Distance between ROARN and IO for spatially-tuned neurons vs non-spatially tuned
sns.set_theme(context="talk", style='white', font_scale=1.2)
plot_wasserstein_distance(data=[spatially_tuned_np_gain, non_spatially_tuned_np_gain], model_names=names_of_interest,
                          nb_seeds=nb_seeds, colors=[spatially_tuned_color, not_spatially_tuned_color],
                          keep_xticks=False, yticks=[0.06, 0.13],
                          filename=f'wasserstein_distance_np_gain_{names_of_interest[0]}_{names_of_interest[1]}_spatially_vs_not', foldername='spatial_receptive_field',
                          title=f"Distribution gap index between\n{names_of_interest[0]} and {names_of_interest[1]}",
                          comparison_labels=neuron_tuning)


# Get vector of the ROARN-IO np gain gap.
filenames = [
            "reg_output/memory-cc-v2_rnn_3_4-k10/0.txt",
            "reg_output/memory-cc-v2_rnn_3_5-k10/0.txt",
            "reg_output/memory-cc-v2_rnn_3_6-k10/0.txt",
            "reg_output/memory-cc-v2_rnn_3_4-Ideal_obs_k10/344.txt",
            "reg_output/memory-cc-v2_rnn_3_5-Ideal_obs_k10/343.txt",
            "reg_output/memory-cc-v2_rnn_3_6-Ideal_obs_k10/383.txt",
            "reg_output/memory-cc-v2_linear_rnn_3_4-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_linear_rnn_3_5-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_linear_rnn_3_6-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_rnn_3_4-k10/344.txt",
            "reg_output/memory-cc-v2_rnn_3_5-k10/343.txt",
            "reg_output/memory-cc-v2_rnn_3_6-k10/383.txt",
        ]
names = ['Untrained ROARN', 'Ideal observer', 'ROARN w/ LR', 'ROARN']
untrained_np_scores = np.array(get_regression_model_specific_performance('LinearSVR', filenames[:3]))
io_np_scores = np.array(get_regression_model_specific_performance('LinearSVR', filenames[3:6]))
rnn_np_scores = np.array(get_regression_model_specific_performance('LinearSVR', filenames[-3:]))
io_gain = (io_np_scores - untrained_np_scores)
rnn_io_gain_gap = (rnn_np_scores - untrained_np_scores) - io_gain
rnn_io_np_gap = rnn_np_scores - io_np_scores
untrained_normalized_rnn_io_gap = (rnn_np_scores/untrained_np_scores) - (io_np_scores/untrained_np_scores)
io_normalized_rnn = (rnn_np_scores/io_np_scores)


####################################################################################################################################
# Same figures as above but for location vs ideal observer
####################################################################################################################################
sns.set_theme(context="talk", style='white', font_scale=1)
names = ['Untrained ROARN', 'Position', 'Ideal observer', 'ROARN w/ LR', 'ROARN']
untrained_spatially_tuned_np = [np.array(spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
untrained_non_spatially_tuned_np = [np.array(not_spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
spatially_tuned_np_gain = []
non_spatially_tuned_np_gain = []
subselection_spatially_tuned_scores = [] # shape: (number of models that was subselected, nb_seeds, number of samples)
subselection_not_spatially_tuned_scores = []
names_of_interest =  ['Position', 'Ideal observer']
colors_of_interest = [position_color, io_color]
for m in names_of_interest:
    subselection_spatially_tuned_scores.append([np.array(spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_spatially_tuned_np_gain = [np.array(spatially_tuned_scores[s][names.index(m)]) - untrained_spatially_tuned_np[s] for s in range(nb_seeds)]
    spatially_tuned_np_gain.append(model_spatially_tuned_np_gain)

    subselection_not_spatially_tuned_scores .append([np.array(not_spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_non_spatially_tuned_np_gain = [np.array(not_spatially_tuned_scores[s][names.index(m)]) - untrained_non_spatially_tuned_np[s] for s in range(nb_seeds)]
    non_spatially_tuned_np_gain.append(model_non_spatially_tuned_np_gain)

subselection_spatially_tuned_scores_flat = [np.concatenate(subselection_spatially_tuned_scores[i])
                                            for i in range(len(subselection_spatially_tuned_scores))]
subselection_not_spatially_tuned_scores_flat =  [np.concatenate(subselection_not_spatially_tuned_scores[i])
                                                 for i in range(len(subselection_not_spatially_tuned_scores))]
if False:
    spatially_tuned_np_gain_flat = [np.concatenate(spatially_tuned_np_gain[i]) for i in range(len(spatially_tuned_np_gain))]
    non_spatially_tuned_np_gain_flat = [np.concatenate(non_spatially_tuned_np_gain[i]) for i in range(len(non_spatially_tuned_np_gain))]
    plot_scatter(title='', x=spatially_tuned_np_gain_flat[1], y=spatially_tuned_np_gain_flat[0],
                 x2=non_spatially_tuned_np_gain_flat[1], y2=non_spatially_tuned_np_gain_flat[0],
                 x_label='Ideal observer\nneural predictivity gain', y_label='Location\nneural predictivity gain',
                 legend=['Spatially-tuned', 'Non-spatially tuned'], save_fig=True, filename='spatially_vs_not_np_gain_loc_io',
                 keep_legend=False)

    plot_scatter(title='', x=subselection_spatially_tuned_scores_flat[1], y=subselection_spatially_tuned_scores_flat[0],
                 x2=subselection_not_spatially_tuned_scores_flat[1], y2=subselection_not_spatially_tuned_scores_flat[0],
                 x_label='Ideal observer\nneural predictivity', y_label='Location\nneural predictivity',
                 legend=['Spatially-tuned neurons', 'Non-spatially tuned'], save_fig=True, filename='spatially_vs_not_np_loc_io')

#FIGURE 2 g
plot_distribution_hist(subselection_spatially_tuned_scores_flat, 'Neural predictivity',
                       'Proportion of neurons (%)', None,
                       'hist_spatial_loc_io_np', discret_hist=False, colors=[position_color, io_color],
                       title='Spatially-tuned neurons', bins=12, yticks=np.arange(0.1, 0.4, 0.1))

#FIGURE 2 h
plot_distribution_hist(subselection_not_spatially_tuned_scores_flat, 'Neural predictivity',
                       'Proportion of neurons (%)', ['Position', 'Ideal observer'],
                       'hist_nonspatial_loc_io_np', discret_hist=False, colors=[position_color, io_color],
                       title='Non-spatially tuned neurons', bins=12, yticks=np.arange(0.1, 0.4, 0.1))


if False:
    # Correlation of raw np across all seeds
    x, y, x2, y2 = spatially_tuned_np_gain_flat[1], spatially_tuned_np_gain_flat[0], non_spatially_tuned_np_gain_flat[1], non_spatially_tuned_np_gain_flat[0]
    without_nan = (np.isnan(x) + np.isnan(y)) == False
    corr, pvalue = pearsonr(np.array(x)[without_nan], np.array(y)[without_nan])
    without_nan = (np.isnan(x2) + np.isnan(y2)) == False
    corr2, pvalue2 = pearsonr(np.array(x2)[without_nan], np.array(y2)[without_nan])
    print(f"Neural predictivity of Loc and IO is more correlated across spatially-tuned neurons "
          f"{corr} than non-spatially tuned {corr2}")

# Correlation of np gain per seeds
spatially_tuned_corrs = []
spatially_tuned_pvalues = []
non_spatially_tuned_corrs = []
non_spatially_tuned_pvalues = []
for s in range(nb_seeds):
    # Correlation between the np gain of Ideal observer and ROARN for each seeds
    # Only np gain for spatially tuned neurons
    without_nan = (np.isnan(spatially_tuned_np_gain[0][s]) + np.isnan(spatially_tuned_np_gain[1][s])) == False
    spatially_tuned_corr, spatially_tuned_pvalue = pearsonr(spatially_tuned_np_gain[0][s][without_nan], spatially_tuned_np_gain[1][s][without_nan])

    without_nan = (np.isnan(non_spatially_tuned_np_gain[0][s]) + np.isnan(non_spatially_tuned_np_gain[1][s])) == False
    non_spatially_tuned_corr, non_spatially_tuned_pvalue = pearsonr(non_spatially_tuned_np_gain[0][s][without_nan],
                                                                    non_spatially_tuned_np_gain[1][s][without_nan])
    spatially_tuned_corrs.append(spatially_tuned_corr)
    spatially_tuned_pvalues.append(spatially_tuned_pvalue)
    non_spatially_tuned_corrs.append(non_spatially_tuned_corr)
    non_spatially_tuned_pvalues.append(non_spatially_tuned_pvalue)
print(f"Neural predictivity of Loc and IO is more correlated across spatially-tuned neurons gain "
      f"{np.mean(spatially_tuned_corrs).round(3)} +-{np.std(spatially_tuned_corrs).round(3)} p={spatially_tuned_pvalues} than non-spatially"
      f" tuned {np.mean(non_spatially_tuned_corrs).round(3)} +-{np.std(non_spatially_tuned_corrs).round(3)} p={non_spatially_tuned_pvalues}")

if False:
    neuron_tuning = ['Spatially-tuned', 'Non-spatially tuned']
    ax = sns.barplot(x=neuron_tuning, y=[np.mean(spatially_tuned_corrs), np.mean(non_spatially_tuned_corrs)],
                     palette=[spatially_tuned_color, not_spatially_tuned_color])
    ax.bar_label(ax.containers[0], fmt='%.2f', padding=8)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    plt.errorbar(x=x_coords, y=y_coords, yerr=[np.std(spatially_tuned_corrs), np.std(non_spatially_tuned_corrs)], fmt="none", c="k", capsize=3)  # , elinewidth=0.5
    sns.set_theme(context="talk", style='white', font_scale=1)
    plt.ylabel(r"Pearson $\rho$")
    x_positions = np.arange(len(neuron_tuning))
    data = [spatially_tuned_corrs, non_spatially_tuned_corrs]
    for i in range(len(x_positions)):
        ax.scatter([x_positions[i]] * len(data[i]), data[i], color='black', s=20, alpha=0.8)
    plt.yticks([], [])
    plt.xticks([], [])
    plt.tight_layout()
    box = ax.get_position() # Make plot thinner
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    sns.despine(top=True, right=True)
    if save_fig:
        os.makedirs(f"{save_fig_folder}/spatial_receptive_field/", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/spatial_receptive_field/correlation_np_gain_loc_io_spatially_vs_not",
                    bbox_inches='tight', dpi=900)
        os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
        plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/correlation_np_gain_loc_io_spatially_vs_not.svg", bbox_inches='tight', format='svg')
        plt.clf()
        plt.close()

#FIGURE appendix fig 2 g-h, show that non-spatial features improve mainly STN and not nSTN
sns.set_theme(context="talk", style='white', font_scale=1.2)
neuron_tuning = ['Spatially-tuned', 'Non-spatially tuned']
plot_wasserstein_distance(data=[spatially_tuned_np_gain, non_spatially_tuned_np_gain], model_names=names_of_interest,
                          nb_seeds=nb_seeds, colors=[spatially_tuned_color, not_spatially_tuned_color],
                          keep_xticks=False, yticks=[0.04, 0.10],
                          filename='wasserstein_distance_np_gain_loc_io_spatially_vs_not', foldername='spatial_receptive_field',
                          title=f"Distribution gap index between\n{names_of_interest[0]} and {names_of_interest[1]}",
                          comparison_labels=neuron_tuning)

#FIGURE appendix fig 2 g-h, show that non-spatial features improve mainly STN and not nSTN
gap_between_loc_and_io_spatially_tuned = [spatially_tuned_np_gain[1][s] - spatially_tuned_np_gain[0][s] for s in range(nb_seeds)]
gap_between_loc_and_ion_nonspatially_tuned = [non_spatially_tuned_np_gain[1][s] - non_spatially_tuned_np_gain[0][s] for s in range(nb_seeds)]
data_flat = np.concatenate([np.concatenate(gap_between_loc_and_io_spatially_tuned), np.concatenate(gap_between_loc_and_ion_nonspatially_tuned)])
df = pd.DataFrame.from_dict({
    'Gap between neural predictivity\ngain of Position and Ideal observer': data_flat,
    'tuning': np.concatenate([
        ['Spatially-tuned' for _ in range(np.concatenate(gap_between_loc_and_io_spatially_tuned).size)],
        ['Non-spatially tuned' for _ in range(np.concatenate(gap_between_loc_and_ion_nonspatially_tuned).size)]
    ]),
    '': ['' for _ in range(data_flat.size)]
})
sns.displot(data=df, x=f"Gap between neural predictivity\ngain of {names_of_interest[0]} and {names_of_interest[1]}",
            hue="tuning", kind="ecdf")
plt.ylabel("Cumulative proportion")
if save_fig:
    plt.savefig(f"{save_fig_folder}/cumulative_distribution_gap_np_gain_loc_io", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/cumulative_distribution_gap_np_gain_loc_io.svg", bbox_inches='tight', dpi=900,
                format='svg')
    plt.clf()
    plt.close()
else:
    plt.show()


####################################################################################################################################
# Same figures as above but for IO vs ROARN w/ LR
####################################################################################################################################
# Add roarn in IO vs ROARN w/ LR figures
sns.set_theme(context="talk", style='white', font_scale=1.2)
names = ['Untrained ROARN', 'Position', 'Ideal observer', 'ROARN w/ LR', 'ROARN']

untrained_spatially_tuned_np = [np.array(spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
untrained_non_spatially_tuned_np = [np.array(not_spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]

spatially_tuned_np_gain = []
non_spatially_tuned_np_gain = []
subselection_spatially_tuned_scores = [] # shape: (number of models that was subselected, nb_seeds, number of samples)
subselection_not_spatially_tuned_scores = []
names_of_interest = ['Ideal observer', 'ROARN w/ LR', 'ROARN']
colors_of_interest = [io_color, linear_color, roarn_color]
for m in names_of_interest:
    subselection_spatially_tuned_scores.append([np.array(spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_spatially_tuned_np_gain = [np.array(spatially_tuned_scores[s][names.index(m)]) - untrained_spatially_tuned_np[s] for s in range(nb_seeds)]
    spatially_tuned_np_gain.append(model_spatially_tuned_np_gain)

    subselection_not_spatially_tuned_scores .append([np.array(not_spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_non_spatially_tuned_np_gain = [np.array(not_spatially_tuned_scores[s][names.index(m)]) - untrained_non_spatially_tuned_np[s] for s in range(nb_seeds)]
    non_spatially_tuned_np_gain.append(model_non_spatially_tuned_np_gain)
spatially_tuned_np_gain_flat = [np.concatenate(spatially_tuned_np_gain[i]) for i in range(len(spatially_tuned_np_gain))]
non_spatially_tuned_np_gain_flat = [np.concatenate(non_spatially_tuned_np_gain[i]) for i in range(len(non_spatially_tuned_np_gain))]
subselection_spatially_tuned_scores_flat = [np.concatenate(subselection_spatially_tuned_scores[i])
                                            for i in range(len(subselection_spatially_tuned_scores))]
subselection_not_spatially_tuned_scores_flat = [np.concatenate(subselection_not_spatially_tuned_scores[i])
                                                for i in range(len(subselection_not_spatially_tuned_scores))]

#FIGURE [version not used] 4 IO and ROARN w/ LR are the same (with ROARN)
plot_distribution_hist(spatially_tuned_np_gain_flat, 'Neural predictivity gain',
                       'Proportion of neurons (%)', None,
                       'hist_spatial_io_roarnlr_roarn_np_gain', discret_hist=False, colors=colors_of_interest,
                       title='Spatially-tuned neurons', bins=10, yticks=np.arange(0.1, 0.4, 0.1))

#FIGURE [version not used] IO and ROARN w/ LR are the same (with ROARN)
plot_distribution_hist(non_spatially_tuned_np_gain_flat, 'Neural predictivity gain',
                       'Proportion of neurons (%)', names_of_interest,
                       'hist_nonspatial_io_roarnlr_roarn_np_gain', discret_hist=False, colors=colors_of_interest,
                       title='Non-spatially tuned neurons', bins=10, yticks=np.arange(0.1, 0.4, 0.1))

# Distance between ROARN w/ LR and ROARN to add to another distance bar plot as a baseline
baseline_distances = []
spatially_tuned_distance = [wasserstein_distance(spatially_tuned_np_gain[1][s], spatially_tuned_np_gain[2][s]) for s in range(nb_seeds)]
baseline_distances.append(np.mean(spatially_tuned_distance))
non_spatially_tuned_distance = []
for s in range(nb_seeds):
    without_nan = (np.isnan(non_spatially_tuned_np_gain[1][s]) + np.isnan(non_spatially_tuned_np_gain[2][s])) == False
    non_spatially_tuned_distance.append(wasserstein_distance(non_spatially_tuned_np_gain[1][s][without_nan], non_spatially_tuned_np_gain[2][s][without_nan]))
baseline_distances.append(np.mean(non_spatially_tuned_distance))
baseline_label = 'Gap index between\nROARN w/ LR\nand ROARN'


sns.set_theme(context="talk", style='white', font_scale=1.2)

names = ['Untrained ROARN', 'Position', 'Ideal observer', 'ROARN w/ LR', 'ROARN']
untrained_spatially_tuned_np = [np.array(spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
untrained_non_spatially_tuned_np = [np.array(not_spatially_tuned_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)]
spatially_tuned_np_gain = []
non_spatially_tuned_np_gain = []
subselection_spatially_tuned_scores = [] # shape: (number of models that was subselected, nb_seeds, number of samples)
subselection_not_spatially_tuned_scores = []
names_of_interest = ['Ideal observer', 'ROARN w/ LR']
colors_of_interest = [io_color, linear_color]
for m in names_of_interest:
    subselection_spatially_tuned_scores.append([np.array(spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_spatially_tuned_np_gain = [np.array(spatially_tuned_scores[s][names.index(m)]) - untrained_spatially_tuned_np[s] for s in range(nb_seeds)]
    spatially_tuned_np_gain.append(model_spatially_tuned_np_gain)

    subselection_not_spatially_tuned_scores .append([np.array(not_spatially_tuned_scores[s][names.index(m)]) for s in range(nb_seeds)])
    model_non_spatially_tuned_np_gain = [np.array(not_spatially_tuned_scores[s][names.index(m)]) - untrained_non_spatially_tuned_np[s] for s in range(nb_seeds)]
    non_spatially_tuned_np_gain.append(model_non_spatially_tuned_np_gain)

#FIGURE maybe to show that io and RAORN w/ LR are the same
spatially_tuned_np_gain_flat = [np.concatenate(spatially_tuned_np_gain[i]) for i in range(len(spatially_tuned_np_gain))]
non_spatially_tuned_np_gain_flat = [np.concatenate(non_spatially_tuned_np_gain[i]) for i in range(len(non_spatially_tuned_np_gain))]
plot_scatter(title='', x=spatially_tuned_np_gain_flat[1], y=spatially_tuned_np_gain_flat[0],
             x2=non_spatially_tuned_np_gain_flat[1], y2=non_spatially_tuned_np_gain_flat[0],
             x_label=f'{names_of_interest[1]}\nneural predictivity gain', y_label=f'{names_of_interest[0]}\nneural predictivity gain',
             legend=['Spatially-tuned', 'Non-spatially tuned'], save_fig=True, filename='spatially_vs_not_np_gain_io_roarnlr', keep_legend=False)

subselection_spatially_tuned_scores_flat = [np.concatenate(subselection_spatially_tuned_scores[i])
                                            for i in range(len(subselection_spatially_tuned_scores))]
subselection_not_spatially_tuned_scores_flat = [np.concatenate(subselection_not_spatially_tuned_scores[i])
                                                for i in range(len(subselection_not_spatially_tuned_scores))]
if False:
    # raw np
    plot_scatter(title='', x=subselection_spatially_tuned_scores_flat[1], y=subselection_spatially_tuned_scores_flat[0],
                 x2=subselection_not_spatially_tuned_scores_flat[1], y2=subselection_not_spatially_tuned_scores_flat[0],
                 x_label=f'{names_of_interest[1]}\nneural predictivity', y_label=f'{names_of_interest[0]}\nneural predictivity',
                 legend=['Spatially-tuned neurons', 'Non-spatially tuned'], save_fig=True, filename='spatially_vs_not_np_io_roarnlr')


#FIGURE 4 IO and ROARN w/ LR are the same
plot_distribution_hist(spatially_tuned_np_gain_flat, 'Neural predictivity gain',
                       'Proportion of neurons (%)', None,
                       'hist_spatial_io_roarnlr_np_gain', discret_hist=False, colors=colors_of_interest,
                       title='Spatially-tuned neurons', bins=10, yticks=np.arange(0.1, 0.4, 0.1))

#FIGURE IO and ROARN w/ LR are the same
plot_distribution_hist(non_spatially_tuned_np_gain_flat, 'Neural predictivity gain',
                       'Proportion of neurons (%)', names_of_interest,
                       'hist_nonspatial_io_roarnlr_np_gain', discret_hist=False, colors=colors_of_interest,
                       title='Non-spatially tuned neurons', bins=10, yticks=np.arange(0.1, 0.4, 0.1))


# Correaltion of raw np across all seeds
x, y, x2, y2 = spatially_tuned_np_gain_flat[1], spatially_tuned_np_gain_flat[0], non_spatially_tuned_np_gain_flat[1], non_spatially_tuned_np_gain_flat[0]
without_nan = (np.isnan(x) + np.isnan(y)) == False
corr, pvalue = pearsonr(np.array(x)[without_nan], np.array(y)[without_nan])
without_nan = (np.isnan(x2) + np.isnan(y2)) == False
corr2, pvalue2 = pearsonr(np.array(x2)[without_nan], np.array(y2)[without_nan])
print(f"Neural predictivity of {names_of_interest[0]} and {names_of_interest[1]} is more correlated across spatially-tuned neurons "
      f"{corr} than non-spatially tuned {corr2}")


# Correlation of np gain per seeds
spatially_tuned_corrs = []
spatially_tuned_pvalues = []
non_spatially_tuned_corrs = []
non_spatially_tuned_pvalues = []
for s in range(nb_seeds):
    # Correlation between the np gain of Ideal observer and ROARN for each seeds
    # Only np gain for spatially tuned neurons
    without_nan = (np.isnan(spatially_tuned_np_gain[0][s]) + np.isnan(spatially_tuned_np_gain[1][s])) == False
    spatially_tuned_corr, spatially_tuned_pvalue = pearsonr(spatially_tuned_np_gain[0][s][without_nan], spatially_tuned_np_gain[1][s][without_nan])

    without_nan = (np.isnan(non_spatially_tuned_np_gain[0][s]) + np.isnan(non_spatially_tuned_np_gain[1][s])) == False
    non_spatially_tuned_corr, non_spatially_tuned_pvalue = pearsonr(non_spatially_tuned_np_gain[0][s][without_nan],
                                                                    non_spatially_tuned_np_gain[1][s][without_nan])
    spatially_tuned_corrs.append(spatially_tuned_corr)
    spatially_tuned_pvalues.append(spatially_tuned_pvalue)
    non_spatially_tuned_corrs.append(non_spatially_tuned_corr)
    non_spatially_tuned_pvalues.append(non_spatially_tuned_pvalue)
print(f"Neural predictivity of {names_of_interest[0]} and {names_of_interest[1]} is more correlated across spatially-tuned neurons gain "
      f"{np.mean(spatially_tuned_corrs).round(3)} +-{np.std(spatially_tuned_corrs).round(3)} p={spatially_tuned_pvalues} than non-spatially"
      f" tuned {np.mean(non_spatially_tuned_corrs).round(3)} +-{np.std(non_spatially_tuned_corrs).round(3)} p={non_spatially_tuned_pvalues}")


#FIGURE io and ROARNLR are the same
neuron_tuning = ['Spatially-tuned', 'Non-spatially tuned']
ax = sns.barplot(x=neuron_tuning, y=[np.mean(spatially_tuned_corrs), np.mean(non_spatially_tuned_corrs)],
                 palette=[spatially_tuned_color, not_spatially_tuned_color]) # sns.set_palette(sns.color_palette("deep"))
ax.bar_label(ax.containers[0], fmt='%.2f', padding=8)
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=[np.std(spatially_tuned_corrs), np.std(non_spatially_tuned_corrs)], fmt="none", c="k", capsize=3)  # , elinewidth=0.5
sns.set_theme(context="talk", style='white', font_scale=1)
plt.ylabel(r"Pearson $\rho$")
x_positions = np.arange(len(neuron_tuning))
data = [spatially_tuned_corrs, non_spatially_tuned_corrs]
for i in range(len(x_positions)):
    ax.scatter([x_positions[i]] * len(data[i]), data[i], color='black', s=20, alpha=0.8)
plt.yticks([], [])
plt.xticks([], [])
plt.tight_layout()
box = ax.get_position() # Make plot thinner
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
sns.despine(top=True, right=True)
if save_fig:
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/correlation_np_gain_loc_io_mem_spatially_vs_not",
                bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/correlation_np_gain_io_roarnlr_spatially_vs_not.svg", bbox_inches='tight', format='svg')
    plt.clf()
    plt.close()


#FIGURE
sns.set_theme(context="talk", style='white', font_scale=1.2)
neuron_tuning = ['Spatially-tuned', 'Non-spatially tuned']
plot_wasserstein_distance(data=[spatially_tuned_np_gain, non_spatially_tuned_np_gain], model_names=names_of_interest,
                          nb_seeds=nb_seeds, colors=[spatially_tuned_color, not_spatially_tuned_color],
                          keep_xticks=False, yticks=[0.01, 0.07, 0.12],
                          filename='wasserstein_distance_np_gain_io_roarnlr_spatially_vs_not', foldername='spatial_receptive_field',
                          title=f"Distribution gap index between\n{names_of_interest[0]} and {names_of_interest[1]}",
                          comparison_labels=neuron_tuning,
                          baseline_value=baseline_distances, baseline_label=baseline_label)


#FIGURE io and ROARNLR are the same, in appendix
gap_between_loc_and_io_mem_spatially_tuned = [spatially_tuned_np_gain[1][s] - spatially_tuned_np_gain[0][s] for s in range(nb_seeds)]
gap_between_loc_and_io_mem_nonspatially_tuned = [non_spatially_tuned_np_gain[1][s] - non_spatially_tuned_np_gain[0][s] for s in range(nb_seeds)]
data_flat = np.concatenate([np.concatenate(gap_between_loc_and_io_mem_spatially_tuned), np.concatenate(gap_between_loc_and_io_mem_nonspatially_tuned)])
df = pd.DataFrame.from_dict({
    f'Gap between neural predictivity\ngain of {names_of_interest[0]} and {names_of_interest[1]}': data_flat,
    'tuning': np.concatenate([
        ['Spatially-tuned' for _ in range(np.concatenate(gap_between_loc_and_io_mem_spatially_tuned).size)],
        ['Non-spatially tuned' for _ in range(np.concatenate(gap_between_loc_and_io_mem_nonspatially_tuned).size)]
    ]),
    '': ['' for _ in range(data_flat.size)]
})
sns.displot(data=df, x=f"Gap between neural predictivity\ngain of {names_of_interest[0]} and {names_of_interest[1]}",
            hue="tuning", kind="ecdf")
plt.ylabel("Cumulative proportion")
if save_fig:
    plt.savefig(f"{save_fig_folder}/cumulative_distribution_gap_np_gain_io_roarnlr", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/spatial_receptive_field/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/spatial_receptive_field/svg/cumulative_distribution_gap_np_gain_io_roarnlr.svg", bbox_inches='tight', dpi=900,
                format='svg')
    plt.clf()
    plt.close()
else:
    plt.show()





####################################################################################################################################
####################################################################################################################################
# Same figures as above but for temporal types instead of spatial tuning
####################################################################################################################################
####################################################################################################################################

#######
# Load temporal profiles
######
temporality_filepath = "rl/figures/feature_importance/23feb25_temporal_classification_mean_v2_p01_distance10/temporal_selectivity_p0.01.pickle"

with open(temporality_filepath, 'rb') as handle:
    temporality = pickle.load(handle)
memory_distance = temporality['memory_distance']
memory_strength = temporality['memory_strength']

neuron_names = temporality["neuron_ids"]
temporal_names = [k for k in temporality.keys() if k not in ["neuron_ids", "memory_strength", "memory_distance"]]
temporal_neurons = [np.array(neuron_names)[temporality[k]] for k in temporal_names]

perceptual_neurons = temporality['perceptual_neurons']
current_trial_memory_neurons = temporality['current_trial_memory_neurons']
long_memory_neurons = temporality['long_memory_neurons']
neuron_types = [perceptual_neurons, current_trial_memory_neurons, long_memory_neurons]


# Get neural predictivity scores per temporal type
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
            "reg_output/memory-cc-v2_linear_rnn_3_4-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_linear_rnn_3_5-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_linear_rnn_3_6-k10-hp21/344.txt",
            "reg_output/memory-cc-v2_rnn_3_4-k10/344.txt",
            "reg_output/memory-cc-v2_rnn_3_5-k10/343.txt",
            "reg_output/memory-cc-v2_rnn_3_6-k10/383.txt",
        ]

names = ['Untrained ROARN', 'Position', 'Ideal observer', 'ROARN w/ LR', 'ROARN']
# Shape: (nb_seeds, number of models i.e. len(names), number of neurons that were classified as this spatial type for this seed)
perceptual_scores = []
near_memory_scores = []
distant_memory_scores = []
scores = [perceptual_scores, near_memory_scores, distant_memory_scores]
group_names = temporal_names
temporal_selectivity  = [perceptual_neurons, current_trial_memory_neurons, long_memory_neurons]
temporal_neurons = [np.array(neuron_names)[perceptual_neurons],
                        np.array(neuron_names)[current_trial_memory_neurons],
                        np.array(neuron_names)[long_memory_neurons]]
for s in range(nb_seeds):
    for temporal_scores, temporal_selection in zip(scores, temporal_neurons):
        # Select files of the 's'-th seed of each model
        subselection_files = np.array(filenames)[np.arange(len(filenames), step=3) + s]
        # Get neural predictivity for each model and accumulate them for each seeds of neuron spatial types classification
        temporal_scores.append(get_regression_model_specific_performance('LinearSVR', subselection_files,
                                                                         None, temporal_selection))

untrained_perceptual_np = []
untrained_near_memory_np = []
untrained_distant_memory_np = []
untrained_temporal_np = [untrained_perceptual_np, untrained_near_memory_np, untrained_distant_memory_np]
for untrained_np, temporal_scores in zip(untrained_temporal_np, scores):
    untrained_np.extend([np.array(temporal_scores[s][names.index('Untrained ROARN')]) for s in range(nb_seeds)])


perceptual_np_gain = []
near_memory_np_gain = []
distant_memory_np_gain = []
temporal_np_gain = [perceptual_np_gain, near_memory_np_gain, distant_memory_np_gain]

subselection_perceptual_scores = []
subselection_near_memory_scores = []
subselection_distant_memory_scores = []
subselection_temporal_scores = [subselection_perceptual_scores, subselection_near_memory_scores, subselection_distant_memory_scores]

names_of_interest = ['ROARN w/ LR', 'ROARN']
colors_of_interest = [linear_color, roarn_color]
for m in names_of_interest:
    for untrained_np, temporal_scores, subselection_scores, np_gain in (
            zip(untrained_temporal_np, scores, subselection_temporal_scores, temporal_np_gain)):
        subselection_scores.append([np.array(temporal_scores[s][names.index(m)]) for s in range(nb_seeds)])
        model_np_gain = [np.array(temporal_scores[s][names.index(m)]) - untrained_np[s] for s in range(nb_seeds)]
        np_gain.append(model_np_gain)
perceptual_np_gain_flat = [np.concatenate(perceptual_np_gain[i]) for i in range(len(perceptual_np_gain))]
near_memory_np_gain_flat = [np.concatenate(near_memory_np_gain[i]) for i in range(len(near_memory_np_gain))]
distant_memory_np_gain_flat = [np.concatenate(distant_memory_np_gain[i]) for i in range(len(distant_memory_np_gain))]

#FIGURE in appendix
plot_scatter(title='', x=perceptual_np_gain_flat[1], y=perceptual_np_gain_flat[0],
             x2=near_memory_np_gain_flat[1], y2=near_memory_np_gain_flat[0],
             x3=distant_memory_np_gain_flat[1], y3=distant_memory_np_gain_flat[0],
             x_label=f'{names_of_interest[1]} neural predictivity gain', y_label=f'{names_of_interest[0]}\nneural predictivity gain',
             legend=['Perceptual', 'Near memory', 'Distant memory'], save_fig=True, filename='temporal_gain_io_rnn',
             keep_legend=True, foldername='temporal_type', colors=temporal_colors)

##########################
# Correlation between IO and ROARN np gain for each temporal type
##########################
x, x2, x3 = perceptual_np_gain_flat[1], near_memory_np_gain_flat[1], distant_memory_np_gain_flat[1]
y, y2, y3 = perceptual_np_gain_flat[0], near_memory_np_gain_flat[0], distant_memory_np_gain_flat[0]
without_nan = (np.isnan(x) + np.isnan(y)) == False
corr, pvalue = pearsonr(np.array(x)[without_nan], np.array(y)[without_nan])
without_nan = (np.isnan(x2) + np.isnan(y2)) == False
corr2, pvalue2 = pearsonr(np.array(x2)[without_nan], np.array(y2)[without_nan])
without_nan = (np.isnan(x3) + np.isnan(y3)) == False
corr3, pvalue3 = pearsonr(np.array(x3)[without_nan], np.array(y3)[without_nan])
print(f"Neural predictivity gain of {names_of_interest[0]} and {names_of_interest[1]} correlation\n"
      f"Perceptual: {corr} {pvalue}; Near memory: {corr2} {pvalue2}; Distant memory: {corr3} {pvalue3};")


# Correlation of np gain per seeds
perceptual_corrs = []
near_memory_corrs = []
distant_memory_corrs = []
temporal_corrs = [perceptual_corrs, near_memory_corrs, distant_memory_corrs]
perceptual_pvalues = []
near_memory_pvalues = []
distant_memory_pvalues = []
temporal_pvalues = [perceptual_pvalues, near_memory_pvalues, distant_memory_pvalues]
for s in range(nb_seeds):
    for np_gain, temporal_corr, temporal_pvalue in zip(temporal_np_gain, temporal_corrs, temporal_pvalues):
        # Correlation between the np gain of Ideal observer and ROARN for each seeds
        # Only np gain for spatially tuned neurons
        without_nan = (np.isnan(np_gain[0][s]) + np.isnan(np_gain[1][s])) == False
        corr, pvalue = pearsonr(np_gain[0][s][without_nan], np_gain[1][s][without_nan])
        temporal_corr.append(corr)
        temporal_pvalue.append(pvalue)

for corr, pval, type in zip(temporal_corrs, temporal_pvalues, temporal_names):
    print(f"Neural predictivity gain of {names_of_interest[0]} and {names_of_interest[1]} for {type}: "
          f"{np.mean(corr).round(2)} +-{np.std(corr).round(2)} p={pval}")

#FIGURE
sns.set_theme(context="talk", style='white', font_scale=1)
temporal_type = ['Perceptual', 'Near memory', 'Distant memory']
ax = sns.barplot(x=temporal_type, y=[np.mean(e) for e in temporal_corrs],
                 palette=temporal_colors) # sns.set_palette(sns.color_palette("deep"))
ax.bar_label(ax.containers[0], fmt='%.2f', padding=8)
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=[np.std(e) for e in temporal_corrs], fmt="none", c="k", capsize=3)  # , elinewidth=0.5
plt.ylabel(r"Pearson $\rho$")
x_positions = np.arange(len(temporal_type))
for i in range(len(x_positions)):
    ax.scatter([x_positions[i]] * len(temporal_corrs[i]), temporal_corrs[i], color='black', s=20, alpha=0.8)
plt.yticks([], [])
plt.xticks([], [])
plt.tight_layout()
box = ax.get_position() # Make plot thinner
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
sns.despine(top=True, right=True)
plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
if save_fig:
    os.makedirs(f"{save_fig_folder}/temporal_type/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/temporal_type/correlation_np_gain_io_rnn_temporal",
                bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/temporal_type/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/temporal_type/svg/correlation_np_gain_io_rnn_temporal.svg", bbox_inches='tight', format='svg')
    plt.clf()
    plt.close()



##############################################################################
# Wasserstein distance between IO and ROARN np gain for each temporal type
##############################################################################
sns.set_theme(context="talk", style='white', font_scale=1)
temporal_categories = ['Perceptual', 'Near memory', 'Distant memory']
#FIGURE
sns.set_theme(context="talk", style='white', font_scale=1.2)
plot_wasserstein_distance(data=temporal_np_gain, model_names=names_of_interest,
                          nb_seeds=nb_seeds, colors=temporal_colors,
                          keep_xticks=False, yticks=[0.08, 0.10, 0.12],
                          filename='wasserstein_distance_np_gain_io_rnn_temporal', foldername='temporal_type',
                          # title=f"Distance between neural predictivity\ngain of {names_of_interest[0]} and {names_of_interest[1]}",
                          # title=f" between\n{names_of_interest[0]} and {names_of_interest[1]}",
                          title=f"Distribution gap index between\n{names_of_interest[0]} and {names_of_interest[1]}",
                          comparison_labels=temporal_categories)



##############################################################################
# Grouped bar plot of np gain
##############################################################################
mean_perceptual_scores = np.zeros((nb_seeds, len(names)))
mean_near_memory_scores = np.zeros((nb_seeds, len(names)))
mean_distant_memory_scores = np.zeros((nb_seeds, len(names)))
mean_scores = [mean_perceptual_scores, mean_near_memory_scores, mean_distant_memory_scores]
median_perceptual_scores = np.zeros((nb_seeds, len(names)))
median_near_memory_scores = np.zeros((nb_seeds, len(names)))
median_distant_memory_scores = np.zeros((nb_seeds, len(names)))
median_scores = [median_perceptual_scores, median_near_memory_scores, median_distant_memory_scores]
for s in range(nb_seeds):
    for model in range(len(names)):
        for temporal_scores, mean_score, median_score in zip(scores, mean_scores, median_scores):
            mean_score[s, model] = np.nanmean(temporal_scores[s][model])
            median_score[s, model] = np.nanmedian(temporal_scores[s][model])

mean_data = np.stack(mean_scores)
median_data = np.stack(median_scores)
group_names = names
bins = ['Perceptual', 'Near memory', 'Distant memory']
group_names2 = ['Untrained ROARN', 'Position', 'Ideal observer', 'ROARN w/ LR', 'ROARN']

#FIGURE
untrained_substracted_mean_data = (mean_data - mean_data[:, :, 0][:, :, None])
grouped_bar_plot(np.transpose(untrained_substracted_mean_data, (2,1,0)), group_names2,
                 bins, error_type='std', xlabel='', ylabel='Neural predictivity gain',
                 filename=f"np_gain_temporal", folder_name='temporal_type',
                 keep_legend=False, rotate_xticks=True, colors=model_colors, yticks=np.arange(0.05, 0.2, 0.05),
                 add_scatterpoints=True)


##############################################################################
# Empirical cumulative distribution function
##############################################################################
# NP gain
# temporal_np_gain = [perceptual_np_gain, near_memory_np_gain, distant_memory_np_gain]
# gap_between_io_and_rnn.shape: (nb temporal type, number of seeds, number of neurons of that temporal type)
gap_between_io_and_rnn = [[] for _  in range(len(temporal_np_gain))]
for np_gain, temporal_gap in zip(temporal_np_gain, gap_between_io_and_rnn):
    # Compute the np gain gap between ROARN and IO for each temporal type (i.e. perceptual, near, distant) for each seeds
    temporal_gap.extend([np_gain[1][s] - np_gain[0][s] for s in range(nb_seeds)])

data_flat_seed = [np.concatenate(e) for e in gap_between_io_and_rnn]
data_flat = np.concatenate(data_flat_seed)
hue_labels = [['Perceptual', 'Near memory', 'Distant memory'][i] for i in range(len(data_flat_seed)) for _ in range(data_flat_seed[i].size)]
df = pd.DataFrame.from_dict({
    f'Gap between neural predictivity\ngain of {names_of_interest[1]} and {names_of_interest[0]}': data_flat,
    'Temporal type': hue_labels,
    '': ['' for _ in range(data_flat.size)]
})
#FIGURE
sns.displot(data=df, x=f"Gap between neural predictivity\ngain of {names_of_interest[1]} and {names_of_interest[0]}", hue="Temporal type", kind="ecdf", palette=temporal_colors)
plt.ylabel("Cumulative proportion")
if save_fig:
    plt.savefig(f"{save_fig_folder}/cumulative_distribution_gap_np_gain_rnn_io_temporal", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/temporal_type/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/temporal_type/svg/cumulative_distribution_gap_np_gain_rnn_io_temporal.svg", bbox_inches='tight', dpi=900,
                format='svg')
    plt.clf()
    plt.close()
else:
    plt.show()


####################################################################################################################################
####################################################################################################################################
# Overlap between spatial and temporal types
####################################################################################################################################
####################################################################################################################################

# FIGURE supplementary Coincident matrix
temporal_categories = ['perceptual', 'near memory', 'distant memory']
avg_grids = [np.zeros(len(neuron_locations) * len(neuron_locations)).reshape((len(neuron_locations), len(neuron_locations))) for s in range(len(temporal_categories))]
for s in range(nb_seeds):
    significant_perceptual_neurons = {n: significant_neurons[s][n] for n in neuron_names[perceptual_neurons]}
    grid = plot_fig2b(significant_perceptual_neurons, neuron_locations, location_labels, file_title=f'perceptual_neurons_seed{s}')
    avg_grids[0] += grid

    significant_intrial_memory_neurons = {n: significant_neurons[s][n] for n in
                                          np.array(neuron_names)[current_trial_memory_neurons]}
    grid = plot_fig2b(significant_intrial_memory_neurons, neuron_locations, location_labels, file_title=f'near_neurons_seed{s}')
    avg_grids[1] += grid

    significant_distant_memory_neurons = {n: significant_neurons[s][n] for n in np.array(neuron_names)[long_memory_neurons]}
    grid = plot_fig2b(significant_distant_memory_neurons, neuron_locations, location_labels, file_title=f'distant_neurons_seed{s}')
    avg_grids[2] += grid

avg_grids = [e/nb_seeds for e in avg_grids]
sns.set_theme(context="talk", style='white', font_scale=1.25)
for i, cat_name in enumerate(temporal_categories):
    plot_coincident_matrix(avg_grids[i], location_labels,title=f'{cat_name} neuron', file_title=f"avg_{cat_name}")

num_perceptual_neurons = len(set(np.array(neuron_names)[perceptual_neurons]))
num_intrial_neurons = len(set(np.array(neuron_names)[current_trial_memory_neurons]))
num_distant_neurons = len(set(np.array(neuron_names)[long_memory_neurons]))
total_num_neurons = num_perceptual_neurons + num_intrial_neurons + num_distant_neurons

significant_perceptual_neurons_excl = [{} for _ in range(nb_seeds)]
significant_intrial_neurons_excl = [{} for _ in range(nb_seeds)]
significant_distant_neurons_excl = [{} for _ in range(nb_seeds)]
num_significant_perceptual_neurons_excl = [0 for _ in range(nb_seeds)]
num_significant_intrial_neurons_excl = [0 for _ in range(nb_seeds)]
num_significant_distant_neurons_excl = [0 for _ in range(nb_seeds)]
num_significant_perceptual_neurons = [0 for _ in range(nb_seeds)]
num_significant_intrial_neurons = [0 for _ in range(nb_seeds)]
num_significant_distant_neurons = [0 for _ in range(nb_seeds)]
num_significant_perceptual_neurons_multiple_fields = [0 for _ in range(nb_seeds)]
num_significant_intrial_neurons_multiple_fields = [0 for _ in range(nb_seeds)]
num_significant_distant_neurons_multiple_fields = [0 for _ in range(nb_seeds)]
for s in range(nb_seeds):
    spatial_selectivity_neurons, sym_spatial_selectivity_neurons, exclusive_sym_spatial_selectivity_neurons = get_selectivity_patterns(
        significant_neurons[s], minigrid_locations=False)
    # Number of STNs per temporal type
    num_significant_perceptual_neurons[s] = np.sum([spatial_selectivity_neurons[n] for n in neuron_names[perceptual_neurons]])
    num_significant_intrial_neurons[s] = np.sum([spatial_selectivity_neurons[n] for n in neuron_names[current_trial_memory_neurons]])
    num_significant_distant_neurons[s] = np.sum([spatial_selectivity_neurons[n] for n in neuron_names[long_memory_neurons]])
    # Number of neurons with multiple spatial resonse fields per temporal type
    num_significant_perceptual_neurons_multiple_fields[s] = np.sum([np.sum(list(significant_neurons[s][n].values())) >= 2 for n in neuron_names[perceptual_neurons]])
    num_significant_intrial_neurons_multiple_fields[s] = np.sum([np.sum(list(significant_neurons[s][n].values())) >= 2 for n in neuron_names[current_trial_memory_neurons]])
    num_significant_distant_neurons_multiple_fields[s] = np.sum([np.sum(list(significant_neurons[s][n].values())) >= 2 for n in neuron_names[long_memory_neurons]])
    # STNs at task-equivalent locations per temporal type
    significant_perceptual_neurons_excl[s] = {n: exclusive_sym_spatial_selectivity_neurons[n] for n in neuron_names[perceptual_neurons]}
    significant_intrial_neurons_excl[s] = {n: exclusive_sym_spatial_selectivity_neurons[n] for n in neuron_names[current_trial_memory_neurons]}
    significant_distant_neurons_excl[s] = {n: exclusive_sym_spatial_selectivity_neurons[n] for n in neuron_names[long_memory_neurons]}
    num_significant_perceptual_neurons_excl[s] = np.sum([np.array(list(significant_perceptual_neurons_excl[s][n].values())).any() for n in significant_perceptual_neurons_excl[s].keys()])
    num_significant_intrial_neurons_excl[s] = np.sum([np.array(list(significant_intrial_neurons_excl[s][n].values())).any() for n in significant_intrial_neurons_excl[s].keys()])
    num_significant_distant_neurons_excl[s] = np.sum([np.array(list(significant_distant_neurons_excl[s][n].values())).any() for n in significant_distant_neurons_excl[s].keys()])
print(f"Total {total_num_neurons} neurons; {num_perceptual_neurons} perceptuals, {num_intrial_neurons} near, {num_distant_neurons} distant")
print(f"Number of STNs for perceptual: {np.mean(num_significant_perceptual_neurons)}, near {np.mean(num_significant_intrial_neurons)}, distant: {np.mean(num_significant_distant_neurons)}")
print(f"Number of STNs with multiple spatial fields: perceptual {np.mean(num_significant_perceptual_neurons_multiple_fields)}, near {np.mean(num_significant_intrial_neurons_multiple_fields)}, distant {np.mean(num_significant_distant_neurons_multiple_fields)}")
print(f"Number of STNs at task-equivalent for perceptual: {np.mean(num_significant_perceptual_neurons_excl)}, near {np.mean(num_significant_intrial_neurons_excl)}, distant: {np.mean(num_significant_distant_neurons_excl)}")

# FIGURE supplementary Plot number of STNs that have multiple spatial field and number of STNs at task-equivalent locations per temporal type
def compute_mean_std(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

categories = ['Perceptual', 'Near\nMemory', 'Distant\nMemory']
total_neurons = [num_perceptual_neurons, num_intrial_neurons, num_distant_neurons]

stn_data = [num_significant_perceptual_neurons, num_significant_intrial_neurons, num_significant_distant_neurons]
multi_field_data = [num_significant_perceptual_neurons_multiple_fields, num_significant_intrial_neurons_multiple_fields, num_significant_distant_neurons_multiple_fields]
task_eq_data = [num_significant_perceptual_neurons_excl, num_significant_intrial_neurons_excl, num_significant_distant_neurons_excl]

# Compute means and standard deviations
stn_means, stn_stds = zip(*[compute_mean_std(data) for data in stn_data])
multi_field_means, multi_field_stds = zip(*[compute_mean_std(data) for data in multi_field_data])
task_eq_means, task_eq_stds = zip(*[compute_mean_std(data) for data in task_eq_data])

sns.set_palette("deep")
sns.set_theme(context="talk", style='white', font_scale=1.5)
fig, ax = plt.subplots(figsize=(10, 9))

# Stacked area plot data
x = np.arange(len(categories))
ax.fill_between(x, 0, total_neurons, color=neuron_color, alpha=0.7, label='All Neurons')
ax.fill_between(x, 0, stn_means, color='#4C72B0', alpha=0.9, label='STN')
ax.fill_between(x, 0, multi_field_means, color='#B5E2FA', alpha=0.9, label='Multiple Spatial Response Fields')
ax.fill_between(x, 0, task_eq_means, color='#009E73', alpha=0.9, label='Task-Equivalent')

# Add error bars
ax.errorbar(x, stn_means, yerr=stn_stds, color='black', alpha=0.5)
ax.errorbar(x, multi_field_means, yerr=multi_field_stds, color='black', alpha=0.5)
ax.errorbar(x, task_eq_means, yerr=task_eq_stds, color='black', alpha=0.5) # capsize=1

ax.set_ylabel('Number of Neurons') # , fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories) # , fontsize=12
ax.legend(frameon=False, loc='upper right') # fontsize=12,

sns.despine()
plt.tight_layout()
plt.savefig(f"{save_fig_folder}/spatial_propeties_per_temporal")
plt.savefig(f"{save_fig_folder}/svg/spatial_propeties_per_temporal.svg", format='svg')
plt.clf()
plt.close()



# FIGURE supplementary
all_firing_rates = []
perceptual_firing_rates = []
near_firing_rates = []
distant_firing_rates = []

for n, data in metadata.items():
    for sublist in data:
        firing_rate = sublist[fr_index]  # Extract firing rate
        all_firing_rates.append(firing_rate)
    # Determine if the neuron is spatially tuned
    if n in neuron_names[perceptual_neurons]:
        perceptual_firing_rates.extend([sublist[fr_index] for sublist in data])
    elif n in neuron_names[current_trial_memory_neurons]:
        near_firing_rates.extend([sublist[fr_index] for sublist in data])
    else:
        distant_firing_rates.extend([sublist[fr_index] for sublist in data])
perceptual_firing_rates = np.array(perceptual_firing_rates)
near_firing_rates = np.array(near_firing_rates)
distant_firing_rates = np.array(distant_firing_rates)

# Plot 1: Percentage of zeros
zero_counts = [np.mean(np.array(d) == 0) for d in
               [perceptual_firing_rates, near_firing_rates, distant_firing_rates]]
bars = plt.bar(['Perceptual', 'Near', 'Distant'], zero_counts,
        color=[perceptual_color, near_color, distant_color])
for bar in bars:
    height = bar.get_height()
    va = 'bottom' if height < 0.7 else 'top'
    y_pos = height - 0.03 if height > 0.7 else height + 0.01
    color = 'black' if height > 0.7 else 'black'
    plt.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{height:.0%}',
             ha='center', va=va,
            # fontsize=11,
             color=color)
plt.ylabel('Silent time steps')
sns.despine()
plt.yticks([])
plt.tight_layout()
plt.show()

# FIGURE supplementary
# Plot 2: Distribution of non-zero values
non_zero_data = []
for d, label, color in zip([perceptual_firing_rates, near_firing_rates, distant_firing_rates],
                          ['Perceptual', 'Near memory', 'Distant memory'], [perceptual_color, near_color, distant_color]):
    non_zero = [x for x in d if x > 0]
    non_zero_data.append(non_zero)
    sns.kdeplot(non_zero, label=label, color=color, log_scale=False)

plt.xlabel('Firing rates (Hz)')
plt.ylabel('Density')
plt.title('Non-silent distribution')
plt.legend(frameon=False)
plt.xlim([-1,37])
sns.despine()
plt.tight_layout()
os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)
plt.savefig(f"{save_fig_folder}/svg/temporal_firing_rates.svg", bbox_inches='tight', format='svg')
plt.close()
plt.clf()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                               gridspec_kw={'height_ratios': [1, 3]})
zero_pcts = [100 * np.mean(np.array(d) == 0) for d in
             [perceptual_firing_rates, near_firing_rates, distant_firing_rates]]
non_zero_pcts = [100 - p for p in zero_pcts]
ax1.bar([0, 1, 2], zero_pcts, color=[perceptual_color, near_color, distant_color])
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(['Perceptual', 'Near', 'Distant'])
ax1.set_ylabel('Silent\ntime steps (%)')
plt.tight_layout()
# Bottom: ECDF of non-zero values
for data, color, label in zip(
        [perceptual_firing_rates, near_firing_rates, distant_firing_rates],
        [perceptual_color, near_color, distant_color],
        ["Perceptual", "Near", "Distant"]):
    non_zero = np.array(data)[np.array(data) > 0]
    x = np.sort(non_zero)
    y = np.arange(1, len(x) + 1) / len(x)
    ax2.plot(x, y, color=color, label=label, lw=2)
ax2.set_xlabel('Non-Zero Firing Rates')
ax2.set_ylabel('ECDF')
ax2.legend()
plt.tight_layout()
plt.show()

#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################


def plot_pie(data, pie_slice_name, title, filename, folder_name=None, colors=None):
    sns.set_context("poster")
    fig, ax = plt.subplots()
    size = 0.3
    if colors is None:
        colors = sns.color_palette("deep", as_cmap=True)
    ax.pie(np.round(data,2), radius=1 - size, colors=colors, wedgeprops=dict(width=1 - size - 0.2, edgecolor='w'), labels=pie_slice_name, autopct='%1.0f%%')
    ax.set(aspect="equal", title=title)
    ax.axis('equal')

    if save_fig:
        if folder_name is not None:
            os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900, pad_inches=0)
            os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
            plt.savefig(f"{save_fig_folder}/svg/{filename}.svg", bbox_inches='tight', format='svg')
        else:
            plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight')
            plt.savefig(f"{save_fig_folder}/{filename}.svg", bbox_inches='tight', format='svg', pad_inches=0)
        plt.clf()
        plt.close()
    else:
        plt.show()
    sns.reset_orig()


##########################
# Pie chart population proportion of each temporal type
##########################
num_perceptual_neurons = len(set(np.array(neuron_names)[perceptual_neurons]))
num_intrial_neurons = len(set(np.array(neuron_names)[current_trial_memory_neurons]))
num_distant_neurons = len(set(np.array(neuron_names)[long_memory_neurons]))
total_num_neurons = num_perceptual_neurons + num_intrial_neurons + num_distant_neurons
perc_perceptual = num_perceptual_neurons / total_num_neurons
perc_intrial = num_intrial_neurons / total_num_neurons
perc_distantmem = num_distant_neurons / total_num_neurons

#FIGURE
plot_pie([perc_perceptual, perc_intrial, perc_distantmem], ['Perceptual', 'Near', 'Distant'],
         '', 'pie_distrib_temporal_types', folder_name='temporal_type', colors=temporal_colors)

overlap_temporal_type_and_spatial = []
for s in range(nb_seeds):
    # Get spatial types of each neurons for this seed
    is_neuron_spatially_tuned, _, _ = get_selectivity_patterns(significant_neurons[s], minigrid_locations=False)
    spatially_tuned_neurons = [n for n in is_neuron_spatially_tuned.keys() if is_neuron_spatially_tuned[n]]
    not_spatially_tuned_neurons = [n for n in is_neuron_spatially_tuned.keys() if not is_neuron_spatially_tuned[n]]

    perc_perceptual_is_spatial = len(set(spatially_tuned_neurons) & set(np.array(neuron_names)[perceptual_neurons])) / len(np.array(neuron_names)[perceptual_neurons])
    perc_intrial_is_spatial= len(set(spatially_tuned_neurons) & set(np.array(neuron_names)[current_trial_memory_neurons])) / len(np.array(neuron_names)[current_trial_memory_neurons])
    perc_distantmem_is_spatial = len(set(spatially_tuned_neurons) & set(np.array(neuron_names)[long_memory_neurons])) / len(np.array(neuron_names)[long_memory_neurons])

    perc_perceptual_is_notspatial = len(set(not_spatially_tuned_neurons) & set(np.array(neuron_names)[perceptual_neurons])) / len(np.array(neuron_names)[perceptual_neurons])
    perc_intrial_is_notspatial= len(set(not_spatially_tuned_neurons) & set(np.array(neuron_names)[current_trial_memory_neurons])) / len(np.array(neuron_names)[current_trial_memory_neurons])
    perc_distantmem_is_notspatial = len(set(not_spatially_tuned_neurons) & set(np.array(neuron_names)[long_memory_neurons])) / len(np.array(neuron_names)[long_memory_neurons])

    overlap_temporal_type_with_spatial_vs_not = [[perc_perceptual_is_spatial, perc_intrial_is_spatial, perc_distantmem_is_spatial],
                                                 [perc_perceptual_is_notspatial, perc_intrial_is_notspatial, perc_distantmem_is_notspatial]]
    overlap_temporal_type_with_spatial_vs_not = (np.array(overlap_temporal_type_with_spatial_vs_not)*100).round()
    overlap_temporal_type_and_spatial.append(overlap_temporal_type_with_spatial_vs_not)

mean_overlap_temporal_type_and_spatial = np.mean(overlap_temporal_type_and_spatial, axis=0)

sns.set_theme(context="talk", style='white', font_scale=1.2)
group_names = ['Perceptual', 'Near\nmemory', 'Distant\nmemory']
fig, ax = plt.subplots()
ax.bar(group_names, mean_overlap_temporal_type_and_spatial[0], color=spatially_tuned_color, edgecolor='none')
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
data = np.array(overlap_temporal_type_and_spatial)[:,0,:].T
plt.errorbar(x=x_coords, y=y_coords, yerr=np.std(data,axis=1), fmt="none", c="k", capsize=3)  # , elinewidth=0.5
ax.bar(group_names, mean_overlap_temporal_type_and_spatial[1], bottom = mean_overlap_temporal_type_and_spatial[0],
       color=not_spatially_tuned_color, edgecolor='none')
x_positions = np.arange(len(group_names))
for i in range(len(x_positions)):
    ax.scatter([x_positions[i]] * len(data[i]), data[i], color='black', s=20, alpha=0.8)
sns.despine(top=True, right=True)
plt.ylabel("Proportion of neurons (%)")
plt.yticks([],[])

for bar in ax.patches:
  ax.text(bar.get_x() + bar.get_width() / 2,
          bar.get_height() / 2 + bar.get_y(),
          round(bar.get_height()), ha = 'center',
          color = 'w', weight = 'bold', size = 16)
if save_fig:
    os.makedirs(f"{save_fig_folder}/overlap_spatial_temporal/", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/overlap_spatial_temporal/bar_distrib_spatial_or_not_over_temporal", bbox_inches='tight', dpi=900)
    os.makedirs(f"{save_fig_folder}/overlap_spatial_temporal/svg", exist_ok=True)
    plt.savefig(f"{save_fig_folder}/overlap_spatial_temporal/svg/bar_distrib_spatial_or_not_over_temporal.svg", bbox_inches='tight', format='svg')
    plt.clf()
    plt.close()
else:
    plt.show()


sns.set_theme(context="talk", style='white', font_scale=1.2)
group_names = ['Perceptual', 'Near', 'Distant']
neuron_subselections = [np.array(neuron_names)[perceptual_neurons],
                        np.array(neuron_names)[current_trial_memory_neurons],
                        np.array(neuron_names)[long_memory_neurons]]
bin_names = ['0', '1', '2', '3', '4']
list_counters = []
for s in range(nb_seeds):
    count_per_type = np.zeros((len(group_names), len(bin_names)))
    for i in range(len(group_names)):
        nb_significant_locations = []
        for n in neuron_subselections[i]:
            nb_significant_locations.append(np.array(list(significant_neurons[s][n].values())).sum())
        count = Counter(nb_significant_locations)
        for j in count.keys():
            count_per_type[i][j] = round(count[j] / sum(count.values()) * 100)
    list_counters.append(count_per_type)


grouped_bar_plot(np.transpose(np.array(list_counters), (1,0,2)), group_names, bin_names, error_type='std',
                 xlabel='Number of spatial response fields', ylabel='Percentage of neurons (%)',
                 filename=f"nb_spatial_receptive_fields_per_temporal_type", folder_name='overlap_spatial_temporal',
                 colors=temporal_colors, rotate_xticks=False)



####################################################################################################################################
####################################################################################################################################
# Measuring spatial selectivity as a continuous variable instead of spatially-tuned or not.
####################################################################################################################################
####################################################################################################################################

with open(temporality_filepath, 'rb') as handle:
    temporality = pickle.load(handle)
neuron_names = temporality["neuron_ids"]
memory_distance = temporality['memory_distance']
memory_strength = temporality['memory_strength']

neuron_mem_distance = {} # Just to be sure that spatial variance and memory distance is correctly aligned for the same neurons
for i in range(neuron_names.size):
    neuron_mem_distance[neuron_names[i]] = memory_distance[i]

# Get all firing rates at each location for each neuron
neurons_fr_per_loc = {}
for neuron_name in metadata.keys():
    if neuron_name not in neurons_fr_per_loc:
        neurons_fr_per_loc[neuron_name] = {}
        for loc in range(1, args.partition_type + 1):
            neurons_fr_per_loc[neuron_name][loc] = []
    for step in metadata[neuron_name]:
        location = step[0][2]
        firing_rate = step[fr_index]
        neurons_fr_per_loc[neuron_name][location].append(firing_rate)


neurons_fr = {}
neuron_high_fr_prob = {}
spatial_variance = {}
for n in neurons_fr_per_loc.keys():
    # Compute the average firing rate of each neuron (across all locations)
    neurons_fr[n] = {}
    neuron_high_fr_prob[n] = {}
    neurons_fr[n]['mean'] = np.concatenate(list(neurons_fr_per_loc[n].values())).mean()
    neurons_fr[n]['std'] = np.concatenate(list(neurons_fr_per_loc[n].values())).std()
    fr_threshold = neurons_fr[n]['mean'] + (3*neurons_fr[n]['std'])
    mean_fr_per_location = []
    # Compute probability of having a high firing rate at each location
    for loc in neurons_fr_per_loc[n]:
        is_high_fr = neurons_fr_per_loc[n][loc] >= fr_threshold
        neuron_high_fr_prob[n][loc] = np.sum(is_high_fr) / is_high_fr.size
        mean_fr_per_location.append(np.mean(neurons_fr_per_loc[n][loc]))
    variance_between_locations = np.var(mean_fr_per_location)
    variance_overall = np.concatenate(list(neurons_fr_per_loc[n].values())).var()
    spatial_variance[n] = variance_between_locations / variance_overall


neurons_nb_spatial_fields = {}
for n in neuron_high_fr_prob.keys():
    neurons_nb_spatial_fields[n] = np.array(list(significant_neurons[0][n].values())).sum()

# Visualize relationship between spatial variance and number of spatial responses fields
#FIGURE
for metric, name_metric in [(spatial_variance, 'Spatial selectivity index')]:
    x = []
    y = []
    for n in metric:
        y.append(metric[n])
        x.append(neurons_nb_spatial_fields[n])
    sns.scatterplot(x=x, y=y, color=neuron_color)
    sns.despine(top=True, right=True)
    plt.xlabel('Number of spatial response fields')
    plt.ylabel(name_metric)
    plt.tight_layout()
    no_nan = ~np.logical_or(np.isnan(y), np.isnan(x))
    corr, _ = pearsonr(np.array(x)[no_nan], np.array(y)[no_nan])
    plt.annotate("$\it{r}$" + f" = {corr:.2f}", (0.01, 0.94), xycoords='axes fraction')
    if name_metric == 'Spatial selectivity index':
        folder_name = 'overlap_spatial_temporal'
        filename = f'{name_metric}and number of spatial response fields'
        if save_fig:
            if folder_name is not None:
                os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900, pad_inches=0)
                os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/svg/{filename}.svg", bbox_inches='tight', dpi=900, format='svg', pad_inches=0)
            else:
                plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', dpi=900)
                plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', format='svg', pad_inches=0)
            plt.clf()
            plt.close()
    else:
        plt.show()
        plt.clf()
        plt.close()
    print(f"Correlation between {name_metric} and number of spatial response fields: {corr}")


# Visualize relationship ROARN - IO np gain and spatial spatial variance
io_filenames = [
            "reg_output/memory-cc-v2_rnn_3_4-Ideal_obs_k10/344.txt",
            "reg_output/memory-cc-v2_rnn_3_5-Ideal_obs_k10/343.txt",
            "reg_output/memory-cc-v2_rnn_3_6-Ideal_obs_k10/383.txt"]
rnn_filenames = [
            "reg_output/memory-cc-v2_rnn_3_4-k10/344.txt",
            "reg_output/memory-cc-v2_rnn_3_5-k10/343.txt",
            "reg_output/memory-cc-v2_rnn_3_6-k10/383.txt"]
untrained_rnn_filenames = [
            "reg_output/memory-cc-v2_rnn_3_4-k10/0.txt",
            "reg_output/memory-cc-v2_rnn_3_5-k10/0.txt",
            "reg_output/memory-cc-v2_rnn_3_6-k10/0.txt"]

np_gap_per_neuron = {} # np_gap_per_neuron[neuron_name] = ROARN score -  IO score
for rnn_fname, io_fname, un_fname in zip(rnn_filenames, io_filenames, untrained_rnn_filenames):
    rnn_np_per_neuron = get_regression_dict(rnn_fname)
    io_np_per_neuron = get_regression_dict(io_fname)
    untrained_np_per_neuron = get_regression_dict(un_fname)
    for n in rnn_np_per_neuron:
        if n not in np_gap_per_neuron:
            np_gap_per_neuron[n] = []
        np_gap_per_neuron[n].append(rnn_np_per_neuron[n]['Corr'][0][1] - io_np_per_neuron[n]['Corr'][0][1])

spatial_selectivity_metric = [(spatial_variance, 'Spatial selectivity index')]
def func(x, a, c, d):
    return a*np.exp(-c*x)+d
from scipy.optimize import curve_fit

#####
# Visualize relationship
#####
with open(temporality_filepath, 'rb') as handle:
    temporality = pickle.load(handle)
neuron_names = temporality["neuron_ids"]
memory_distance = temporality['memory_distance']
memory_strength = temporality['memory_strength']

neuron_mem_distance = {} # Just to be sure that it is correctly aligned for the same neurons
for i in range(neuron_names.size):
    neuron_mem_distance[neuron_names[i]] = memory_distance[i]

#FIGURE
spatial_selectivity_metric = [(spatial_variance, 'Spatial selectivity index')]
for metric, name_metric in spatial_selectivity_metric:
    sns.set_theme(context="talk", style='white', font_scale=1.2)
    x = []
    y = []
    c = []
    for n in neuron_mem_distance:
        np_gap = np.mean(np_gap_per_neuron[n])
        if ~np.logical_or(np.isnan(metric[n]), np.isnan(np_gap)):
            x.append(metric[n])
            y.append(neuron_mem_distance[n])
    sns.scatterplot(x=x, y=y, color=neuron_color)
    plt.xlabel(name_metric)
    plt.ylabel('Memory distance')
    ax = plt.gca()


    popt, pcov = curve_fit(func, x, y, p0=(1, 1e-6, 1))
    curve_y = func(np.array(x), *popt)
    sorted_pairs = sorted(zip(x, curve_y))
    x_sorted, y_sorted = zip(*sorted_pairs)
    # Convert back to lists
    x_sorted = list(x_sorted)
    y_sorted = list(y_sorted)
    plt.plot(x_sorted, y_sorted, '--', c='black', alpha=0.7, label='Exponential fit')
    plt.legend()

    no_nan = ~np.logical_or(np.isnan(curve_y), np.isnan(x))
    corr, _ = pearsonr(np.array(curve_y)[no_nan], np.array(y)[no_nan])

    plt.annotate("$\it{r}$" + f" = {corr:.2f}", (0.8, 0.80), xycoords='axes fraction')
    plt.tight_layout()
    sns.despine(top=True, right=True)
    if name_metric == 'Spatial selectivity index':
        folder_name = 'overlap_spatial_temporal'
        filename = f'{name_metric}_exponential_fit'
        if save_fig:
            if folder_name is not None:
                os.makedirs(f"{save_fig_folder}/{folder_name}/", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/{folder_name}/{filename}", bbox_inches='tight', dpi=900)
                os.makedirs(f"{save_fig_folder}/svg", exist_ok=True)
                plt.savefig(f"{save_fig_folder}/svg/{filename}.svg", bbox_inches='tight', dpi=900, format='svg')
            else:
                plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', dpi=900)
                plt.savefig(f"{save_fig_folder}/{filename}", bbox_inches='tight', format='svg')
            plt.clf()
            plt.close()
    else:
        plt.show()
        plt.clf()
        plt.close()
    print(f"Correlation between {name_metric} and memory distance: {pearsonr(np.array(x)[no_nan], np.array(y)[no_nan])}")





