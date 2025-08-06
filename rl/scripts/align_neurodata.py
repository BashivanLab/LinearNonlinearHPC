import scipy.io as sio
import h5py


"""
##########################################################################
# Get neuronal data in the same format as the activations from align_activations.py

Examples:
neurodata[unit_id][session_id][trial_id]['locations_9']: [1, 3, 4, 5, 6, 7, 9] # Locations of monkey when the maze in partitioned in 9 allocentric regions
neurodata[unit_id][session_id][trial_id]['firing_rates_9']: [0.0, 0.5, 0.0, 23.3, 12.3, 32.4, 2.0] #  Firing rate of neuron 'unit_id' when at the locations in 'location_9' during 'trial_id' (which is during 'session_id')
neurodata.attrs['location_9_names'] = ['Goal NW', 'Goal NE', 'Dec N', 'Corr N', 'Center', 'Corr S', 'Dec S', 'Goal SW', 'Goal SE']
               
9_loc partitions the maze into 9 allocentric locations
The firing rate is averaged for each of those locations

##########################################################################
"""


matfiles = ['scripts/temp/neurodata_firing_rate_per_location/InputCells_XMaze_5Direc.mat',
            'scripts/temp/neurodata_firing_rate_per_location/InputCells_XMaze_9XY.mat']

h5_file_path = 'aligned_hpc_neurons.h5'
hf = h5py.File(h5_file_path, 'a')
location_partitioning_type = ['_5', '_9']

for k in range(len(matfiles)):
    mat_content = sio.loadmat(matfiles[k], squeeze_me=True, struct_as_record=False)
    for i in range(len(mat_content['InputCells'].Session)):
        unit_id = mat_content['InputCells'].UnitIDs[i]
        session_id = mat_content['InputCells'].Session[i]
        trial_ids = mat_content['InputCells'].TrialIDs[i]
        firing_rate_per_trial = {}
        locations_per_trial = {}

        for j in range(len(trial_ids)):
            if not trial_ids[j] in firing_rate_per_trial.keys():
                firing_rate_per_trial[trial_ids[j]] = [mat_content['InputCells'].Signal[i][j]]
                locations_per_trial[trial_ids[j]] = [int(mat_content['InputCells'].Condition[i][j])]
            else:
                firing_rate_per_trial[trial_ids[j]].append(mat_content['InputCells'].Signal[i][j])
                locations_per_trial[trial_ids[j]].append(mat_content['InputCells'].Condition[i][j])

        for trial_id in firing_rate_per_trial.keys():
            dataset_path = f"{unit_id}/{session_id}/{trial_id}"
            hf.create_dataset(f"{dataset_path}/locations{location_partitioning_type[k]}", data=locations_per_trial[trial_id])
            hf.create_dataset(f"{dataset_path}/firing_rates{location_partitioning_type[k]}", data=firing_rate_per_trial[trial_id])

    hf.attrs[f"location{location_partitioning_type[k]}_names"] = mat_content['InputCells'].CondLabels

hf.close()


