from glob import glob
import scipy.io as sio
import h5py
import numpy as np
import pickle

if __name__ == "__main__":
    location_type = 9

    # Extracts info from monkey MATLAB files (xmaze.mat)
    extract_from_matlab_files = False
    if extract_from_matlab_files:
        root_folder = "../monkey-data/Results/"
        monkey_folder = ['Raul/', 'Woody/']
        sessions = glob(root_folder + monkey_folder[0] + "*/*_XMaze.mat", recursive=True)
        raul_session_index = (0, len(sessions))
        sessions.extend(glob(root_folder + monkey_folder[1] + "*/*_XMaze.mat", recursive=True))
        sessions.sort()

        # Extract session information from each matlab file
        session_info = {}
        for s in sessions:
            mat_contents = sio.loadmat(s, squeeze_me=True, struct_as_record=False)
            session_name = ''.join([s.split('/')[-3][0], s.split('/')[-2]])
            session_info[session_name] = {'trial': mat_contents['XMazeStruct'].TrialSummary[1:, 0].tolist(),
                                          'direction': mat_contents['XMazeStruct'].TrialSummary[1:, 1].tolist(),
                                          'context': mat_contents['XMazeStruct'].TrialSummary[1:, 3].tolist(),
                                          'colors_west': mat_contents['XMazeStruct'].TrialSummary[1:, 4].tolist(),
                                          'colors_east': mat_contents['XMazeStruct'].TrialSummary[1:, 5].tolist(),
                                          'outcome': mat_contents['XMazeStruct'].TrialSummary[1:, 6].tolist(),
                                          'wood_setting': mat_contents['XMazeStruct'].SessionInfo.Context1NParams[1].tolist(), # Best color to worst
            }


    print('.')

    if extract_from_matlab_files:
        pickle_out = open("temp_debug_neurodata.pickle", "wb")
        pickle.dump(session_info, pickle_out)
        pickle_out.close()

    if not extract_from_matlab_files:
        pickle_in = open("temp_debug_neurodata.pickle", "rb")
        session_info = pickle.load(pickle_in)
        pickle_in.close()

    def get_choice_and_reward_value(session_info, s, t):
        """
        return:
        2 if picked the object giving the highest reward
        1 if received the medium reward
        0 if received the worst
        """
        wood_setting = session_info[s]['wood_setting']
        outcome = session_info[s]['outcome'][t]
        context = session_info[s]['context'][t]
        color_west = session_info[s]['colors_west'][t]
        colors_east = session_info[s]['colors_east'][t]
        west_is_biggest_index = wood_setting.index(color_west) > wood_setting.index(colors_east)
        get_object_picked = { # key = (Outcome, context, west_is_biggest_index)
            # Picked the object with the smallest index
            ('Correct', 'Wood', True): colors_east, # west has the biggest index, thus east is the smallest index
            ('Correct', 'Wood', False): color_west,
            # Picked the object with the biggest index
            ('Correct', 'Steel', True): color_west,
            ('Correct', 'Steel', False): colors_east,
            # Picked the object with the biggest index
            ('Incorrect', 'Wood', True): color_west,
            ('Incorrect', 'Wood', False): colors_east,
            # Picked the object with the smallest index
            ('Incorrect', 'Steel', True): colors_east,
            ('Incorrect', 'Steel', False): color_west,
        }
        object_picked = get_object_picked[(outcome, context, west_is_biggest_index)]
        if context == 'Steel':
            # Encode picked objects from 0 to 2. Use wood_setting to determine which integer is associated to each color just for convenience
            # Reward value: Highest reward == index 2, medium reward == index 1, lowest
            return wood_setting.index(object_picked), wood_setting.index(object_picked)
        else:
            return wood_setting.index(object_picked), wood_setting[::-1].index(object_picked)


    # Load InputCells to have locations
    h5_neurodata_path = 'rl/aligned_hpc_neurons.h5'
    neurodata = h5py.File(h5_neurodata_path, 'r')

    # For each neuron (keys), metadata matrix where each row is a sample (session_id, trial_id, location) (in temporal order) and each column is a feature (e.g. location, reward, firing rate, etc.)
    metadata = {}

    for neuron_id in neurodata.keys():
        session_id = list(neurodata[neuron_id].keys())[0]
        metadata[neuron_id] = []
        for trial_id in neurodata[neuron_id][session_id].keys():
            trial_i = session_info[session_id]['trial'].index(trial_id)
            wood_setting = session_info[session_id]['wood_setting']
            direction = session_info[session_id]['direction'][trial_i]
            unique_locations = np.unique(neurodata[neuron_id][session_id][trial_id][f"locations_{location_type}"][:])
            if direction == 'South':
                # Monkey is going from north to south. Thus, the locations occured in an Ascending order
                # i.e. the start location is either 1 or 2, then the locations are 3,4,5,6,7, and finally 8 or 9
                trial_locations = sorted(unique_locations)
                context_visible = [4, 5, 6, 7, 8, 9]
                goals_visible = [7, 8, 9]
                end_arm = [8, 9]
            elif direction == 'North':
                # Going from South to North. locations occurred in descending order
                # i.e. the start location is either 9 or 8, then the locations are 7, 6, 5, 4, 3 and finally 2 or 1
                trial_locations = sorted(unique_locations, reverse=True)
                context_visible = [1, 2, 3, 4, 5, 6]
                goals_visible = [1, 2, 3]
                end_arm = [1, 2]
            else:
                print('Error at', session_id, neuron_id, trial_id, 'invalid direction:', direction)

            for loc in trial_locations:
                # one hot every factors, 0s if not available
                l = np.eye(9)[int(loc)-1]
                dir_onehot = np.eye(2)[[0 if direction == 'South' else 1][0]]

                if loc in context_visible:
                    if session_info[session_id]['context'][trial_i] == 'Wood':
                        context = np.eye(2)[1]
                    elif session_info[session_id]['context'][trial_i] == 'Steel':
                        context = np.eye(2)[0]
                else:
                    context = np.zeros(2)

                if loc in goals_visible:
                    if not session_info[session_id]['colors_west'][trial_i] in wood_setting or not session_info[session_id]['colors_east'][trial_i] in wood_setting:
                        # happens rarely, but for some trials, a goal object has a color that shouldn't appear in the current setting.
                        continue
                    # encode object based on wood_setting, i.e. if wood_setting = Red > Blue > Green
                    # encode the Red object as: [1,0,0], Blue: [0,1,0], and Green = [0,0,1]
                    object_west = np.eye(3)[wood_setting.index(session_info[session_id]['colors_west'][trial_i])]
                    object_east = np.eye(3)[wood_setting.index(session_info[session_id]['colors_east'][trial_i])]
                else:
                    object_west = np.zeros(3)
                    object_east = np.zeros(3)

                if loc in end_arm:
                    if session_info[session_id]['outcome'][trial_i] == 'Correct':
                        choice_optimality = np.eye(2)[1]
                    elif session_info[session_id]['outcome'][trial_i] == 'Incorrect':
                        choice_optimality = np.eye(2)[0]
                    else:
                        # Sometimes outcome is userStopped or Time run out
                        raise NotImplementedError
                    # one hot encoding where if picked high value object = [1,0,0], mid value = [0,1,0], and low value = [0,0,1])
                    object_chosen_int, reward_value_int = get_choice_and_reward_value(session_info, session_id, trial_i)
                    reward = np.eye(3)[reward_value_int]
                    object_chosen = np.eye(3)[object_chosen_int]
                else:
                    choice_optimality = np.zeros(2)
                    reward = np.zeros(3)
                    object_chosen = np.zeros(3)

                index, = np.where(neurodata[neuron_id][session_id][trial_id][f"locations_{location_type}"][:] == loc)
                # If there is multiple instance with the same sample id, i.e. repeat a location in a trial, use the last instance
                firing_rate = neurodata[neuron_id][session_id][trial_id][f"firing_rates_{location_type}"][index[-1]]
                sample_id = (session_id, trial_id, loc)
                step_input = [[sample_id], [firing_rate], l, dir_onehot, context, object_west, object_east, object_chosen, reward, choice_optimality]
                flat_step_input = [e for sublist in step_input for e in sublist]
                metadata[neuron_id].append(flat_step_input)

    print('Done creating metadata matrices')

    save_metadata_matrix = True
    if save_metadata_matrix:
        pickle_out = open(f"metadata.pickle", "wb")
        pickle.dump(metadata, pickle_out)
        pickle_out.close()

"""
To separate sample_ids, firing_rate, and meta-information
"""
def get_only_meta_info(metadata, neuron_id, include_firing_rate=False):
    if include_firing_rate:
        meta_info_only = [e[1:] for e in metadata[neuron_id]]
    else:
        meta_info_only = [e[2:] for e in metadata[neuron_id]]
    return meta_info_only

def get_only_firing_rate(metadata, neuron_id):
    ff_only = [e[1] for e in metadata[neuron_id]]
    return ff_only

def get_only_sample_id(metadata, neuron_id):
    sample_id_only = [e[0] for e in metadata[neuron_id]]
    return sample_id_only


def align_metadata_and_activation(metadata, activations, neuron_id):
    """
    return:
    list of metadata and list of activation where each row is the same sample (i.e., (session_id, trial_id, location))
    metadata first column contains the sample ids
    """
    activation_sample_ids = []
    session_id = neuron_id.split('_')[0]
    for trial_id in activations[session_id].keys():
        for location in activations[session_id][trial_id].keys():
            activation_sample_ids.append((session_id, trial_id, int(location)))

    metadata_sample_ids = get_only_sample_id(metadata, neuron_id)
    samples_to_remove_from_metadata = set(metadata_sample_ids) - set(activation_sample_ids)
    aligned_metadata = [e for e in metadata[neuron_id] if not e[0] in samples_to_remove_from_metadata]

    aligned_metadata_sample_ids = [e[0] for e in aligned_metadata]
    aligned_activations = []
    for ids in aligned_metadata_sample_ids:
        s, t, l = ids[0], ids[1], str(ids[2])
        aligned_activations.append(activations[s][t][l][:])

    # Check that everything is fine
    ids = set()
    for e in aligned_metadata_sample_ids:
        if e in ids:
            print('duplicated sample id:', e)
            raise NotImplementedError
        else:
            ids.add(e)
    assert len(aligned_metadata) == len(aligned_activations)
    return aligned_metadata, aligned_activations
