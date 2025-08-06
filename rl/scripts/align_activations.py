import numpy as np
import argparse
import h5py
import os
from pathlib import Path
import copy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import collections

# Creating 9XY field and averaging activation for each combination
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model-type", default=None,
                    help="type of model to use (default from model.py)")
parser.add_argument("--checkpoint", default=None,
                    help="Checkpoint id if want to select a specific ones. If none, select the final checkpoint")
args = parser.parse_args()
checkpoint_id = args.checkpoint
plot_distributions = False

if args.model_type is not None and 'epn' in args.model_type:
    inspected_layer = 'max'
    layer_name = 'max_layer'
elif args.model_type == 'linear_rnn' or args.model_type == 'hidden_state':
    inspected_layer = 'memory_rnn_hidden_state'
    layer_name = 'hidden_state'
else:
    inspected_layer = 'memory_rnn_cell_state'
    layer_name = 'cell_state'

# if checkpoint, path to checkpoint folder
if checkpoint_id is not None:
    h5_file_path = f'activations_{args.model}_{args.env}/activations_{args.model}_{args.env}_{checkpoint_id}.h5'
else:
    h5_file_path = f'activations_{args.model}_{args.env}.h5'

# Check if local or on CC
save_fig_folder = "rl/figures"
if checkpoint_id is not None:
    os.makedirs(f'activations_{args.model}_{args.env}/', exist_ok=True)


hf = h5py.File(h5_file_path, 'r+')
activations = hf['layer_activations']
env_data = hf['env_data']

if f'aligned_activations_{layer_name}' in hf:
    # if already exist, check if it is in the wrong session_id name format and need to redo it
    print(list(hf[f'aligned_activations_{layer_name}'].keys())[0])
    if '\\' in list(hf[f'aligned_activations_{layer_name}'].keys())[0]:
        del hf[f'aligned_activations_{layer_name}']
    else:
        raise Exception(f"aligned_activations_{layer_name} already in {h5_file_path}")

LocLabels_9XY = {
    (1, 4): 9,  # Goal SE
    (1, 2): 8,  # Goal SW
    (1, 3): 7,  # Dec S
    (2, 3): 6,  # Corr S
    (3, 3): 5,  # Center
    (4, 3): 4,  # Corr N
    (5, 3): 3,  # Dec N
    (5, 4): 2,  # Goal NE
    (5, 2): 1,  # Goal NW
}

# format session_id to R2014XXXX OR W2014XXXX
def format_session_id(session_id):
    if len(session_id) != 9:
        return str(session_id)[2] + str(session_id).split('\\\\')[-1][:-1]
    else:
        return session_id

# if 'session_id' not in env_data.keys():
dt = h5py.string_dtype(encoding='utf-8')
formatted_session_ids = np.array([format_session_id(session_id) for session_id in env_data['session_id']], dtype=dt)
env_data['session_id'][:] = formatted_session_ids


# Create a new field condition_9XY
def create_condition_9XY(agent_locations, location_mapping):
    converted_locations = []
    for loc in agent_locations:
        tuple_loc = tuple(loc)
        if tuple_loc in location_mapping:
            converted_locations.append(location_mapping[tuple_loc])
        else:
            converted_locations.append('invalid location')
    return converted_locations

# env_data['9XY_condition'][:] = create_condition_9XY(env_data['agent_location'][:], LocLabels_9XY)
chunk_size = 10000
if '9XY_condition' not in env_data.keys():
    env_data.create_dataset('9XY_condition', data=create_condition_9XY(env_data['agent_location'][0:chunk_size], LocLabels_9XY), compression="gzip", chunks=True, maxshape=(None,))
    for i in range(chunk_size, env_data['agent_location'].shape[0], chunk_size):
        locs = create_condition_9XY(env_data['agent_location'][i:i+chunk_size], LocLabels_9XY)
        hf['env_data/9XY_condition'].resize((hf['env_data/9XY_condition'].shape[0] + np.array(locs).shape[0]), axis=0)
        hf['env_data/9XY_condition'][-np.array(locs).shape[0]:] = locs

if plot_distributions:
    # Plot labels distribution
    location_counter = collections.Counter(env_data['9XY_condition'][:])
    locations = sorted(list(location_counter.keys()))
    location_frequency = [location_counter[e] for e in locations]
    fig, ax = plt.subplots()
    bars = ax.bar(x=locations, height=location_frequency)
    ax.bar_label(bars)
    ax.set_xticks(locations)
    plt.title('Number of steps at each locations')
    plt.xlabel('Locations')
    plt.ylabel('Number of occurrences')
    plt.savefig(f"{save_fig_folder}/9loc_labels_raw_distribution_{args.model}_{args.env}.png")
    plt.clf()

# Create a new field condition_5Direc
def create_condition_5Direc(conditions, trial_ids):
    # Initialize condition_5 list with 'invalid location'
    condition_5 = ['invalid location'] * len(conditions)

    # Create sets for different conditions
    set_1_5 = {1, 2, 8, 9}  # Arm-Branch/Arm-End
    set_2_4 = {3, 7}  # Branch-Corr/Branch-Arm
    set_3 = {4, 5, 6}  # Corr

    # Initialize previous_valid_trial_id, previous_valid_condition, and previous_condition_5 variables
    previous_valid_trial_id = None
    previous_valid_condition = None
    previous_condition_5 = None

    # Loop through conditions and trial_ids
    for i, (condition, trial_id) in enumerate(zip(conditions, trial_ids)):
        if condition == 'invalid location':
            continue

        if i > 0 and condition == previous_valid_condition:
            condition_5[i] = previous_condition_5

        elif condition in set_1_5:
            # Set condition_5 value to 1 if it's the first non-invalid condition or trial_id has changed
            if i == 0 or trial_id != previous_valid_trial_id:
                condition_5[i] = 1
            else:
                # Find the next valid trial_id
                next_valid_trial_id = trial_id
                for next_condition, next_trial_id in zip(conditions[i + 1:], trial_ids[i + 1:]):
                    if next_condition != 'invalid location':
                        next_valid_trial_id = next_trial_id
                        break

                # Set condition_5 value to 5 if next_valid_trial_id is different from current trial_id
                if next_valid_trial_id != trial_id:
                    condition_5[i] = 5
                else:
                    condition_5[i] = 'invalid location'

            # Update previous_valid_trial_id, previous_valid_condition, and previous_condition_5
            previous_valid_trial_id = trial_id
            previous_valid_condition = condition
            previous_condition_5 = condition_5[i]

        elif condition in set_2_4:
            # Set condition_5 value to 2 if previous_valid_condition is in set_1_5 and trial_id is the same
            if previous_valid_condition in set_1_5 and trial_id == previous_valid_trial_id:
                condition_5[i] = 2
            else:
                condition_5[i] = 4

            # Update previous_valid_trial_id, previous_valid_condition, and previous_condition_5
            previous_valid_trial_id = trial_id
            previous_valid_condition = condition
            previous_condition_5 = condition_5[i]

        elif condition in set_3:
            # Set condition_5 value to 3
            condition_5[i] = 3

            # Update previous_valid_trial_id, previous_valid_condition, and previous_condition_5
            previous_valid_trial_id = trial_id
            previous_valid_condition = condition
            previous_condition_5 = condition_5[i]

    return condition_5


# Initialize aligned_activation dictionary
aligned_activation = defaultdict(list)

# Initialize a set to store unique (session_id, trial_id, 9XY_condition) combinations
unique_combinations = set()
# duplicate_combination = []

#  collect unique (session_id, trial_id, 9XY_condition) combinations
for i, activation in enumerate(activations[inspected_layer]):
    # Get the corresponding session_id, trial_id, and 9XY_condition for each activation
    current_session_id = env_data['session_id'][i]
    current_trial_id = env_data['trial_id'][i]
    current_condition_9 = env_data['9XY_condition'][i]

    # Skip 'invalid location'
    if current_condition_9 == 'invalid location':
        continue

    unique_combinations.add((current_session_id, current_trial_id, current_condition_9))

    if (current_session_id, current_trial_id, current_condition_9) not in unique_combinations:
        continue

    # Store the activations in the aligned_activation dictionary based on the current combination
    aligned_activation[(current_session_id, current_trial_id, current_condition_9)].append(activation)


# Compute the average activation for each (session_id, trial_id, condition_9) combination
for key in aligned_activation:
    # Compute the average along axis 0, taking into account the shape (n, 1, 256)
    aligned_activation[key] = np.mean(np.vstack(aligned_activation[key]), axis=0)

if plot_distributions:
    # Plot labels distribution after averaging activation with the same ids (session, trial, location)
    locs = [e[2] for e in list(aligned_activation.keys())]
    location_counter = collections.Counter(locs)
    locations = sorted(list(location_counter.keys()))
    location_frequency = [location_counter[e] for e in locations]
    fig, ax = plt.subplots()
    bars = ax.bar(x=locations, height=location_frequency)
    ax.bar_label(bars)
    ax.set_xticks(locations)
    plt.title('Number of activations at each locations')
    plt.xlabel('Locations')
    plt.ylabel('Number of activations')
    plt.savefig(f"{save_fig_folder}/9loc_activations_distribution_{args.model}_{args.env}.png")
    plt.clf()

# Add aligned_activation to the all_activation dictionary
print('Adding aligned_activations in hdf5 file')
hf_aligned_activations = hf.require_group(f'aligned_activations_{layer_name}')
for k in aligned_activation.keys():
    dataset_path = '/'.join([str(k[0], 'utf-8'), str(k[1], 'utf-8'), str(k[2])])
    hf_aligned_activations.create_dataset(dataset_path, data=aligned_activation[k], compression="gzip", chunks=True, maxshape=(None,))
hf.close()
print(f'Done adding aligned_activations_{layer_name} in {h5_file_path}')