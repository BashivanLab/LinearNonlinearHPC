import os
import argparse
from datetime import date
import ast
import time
from random import shuffle
from random import sample
import itertools
from scripts.create_metadata_matrix import align_metadata_and_activation
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.svm import LinearSVR
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model-type", default=None,
                    help="type of model to use (default from model.py)")
parser.add_argument("--exp-name", default=None,
                    help="Name of the experiment to identify the output files")
parser.add_argument("--checkpoint", default=None,
                    help="Checkpoint id if want to select a specific ones. If none, look for hdf5 files without a checkpoint id")
parser.add_argument("--nb-folds", default=10,
                    help="In how many folds to split the train/test sets. 10 means 1/10 (10%) of the dataset is used as test set."
                         "5 means 20%.")
args = parser.parse_args()

checkpoint_id = args.checkpoint
nb_folds = int(args.nb_folds)
experiment_name = args.exp_name
sample_index = 0
fr_index = 1
factors_start_index = 2
plot_fig = False
save_fig = False

pca_dim = None
# only_location = False
# n_steps = 1
is_meta_reg = False
reg_name = 'LinearSVR'
reg_type = LinearSVR
hyperparam = {'reg__intercept_scaling': 3.0, 'reg__C': 0.001, 'reg__max_iter': 100000, 'reg__loss': 'squared_epsilon_insensitive', 'reg__dual': False}



if checkpoint_id is not None:
    h5_file_path = f'activations_{args.model}_{args.env}/activations_{args.model}_{args.env}_{checkpoint_id}.h5'
else:
    h5_file_path = f'activations_{args.model}_{args.env}.h5'

# Check if local or on CC
if os.path.exists('/home/daiglema/scratch/'):
    h5_file_path = '/home/daiglema/scratch/' + h5_file_path
    save_fig_folder = '/home/daiglema/2d-memory-monkeytrials/rl/figures/regression'
    folder_path = '/home/daiglema/2d-memory-monkeytrials/rl/'
else:
    save_fig_folder = "/Users/maxim/PycharmProjects/2d-memory/rl/figures/regression"
    folder_path = "/Users/maxim/PycharmProjects/2d-memory/rl/"

os.makedirs(f"{save_fig_folder}/{experiment_name}/", exist_ok=True)

if args.model_type is not None and 'epn' in args.model_type:
    layer_name = 'max_layer'
elif args.model_type == 'linear_rnn' or args.model_type == 'hidden_state':
    layer_name = 'hidden_state'
else:
    layer_name = 'cell_state'

# Load activations
try:
    hf = h5py.File(h5_file_path, 'r')
except:
    raise FileNotFoundError
activations = hf[f"aligned_activations_{layer_name}"]
pickle_in = open('metadata.pickle', "rb")
metadata = pickle.load(pickle_in)
pickle_in.close()


def get_regression_params(lines, wanted_neuron, regression_model):
    for line in lines:
        neuron = line[8:line.find(", Regressor: ")]
        if wanted_neuron == neuron:
            reg_model = line[line.find("Regressor: ") + len("Regressor: "):line.find(", Best_result:")]
            if reg_model == regression_model:
                params_string = line[line.find("Param: ") + len("Param: "):line.find(", Runtime:")]
                params = ast.literal_eval(params_string)
                return params


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


def get_meta_dataset(neuron_metadata, firing_index, factors_start_index, add_multiplicative_factors=False, delete_n_steps_overlap=False, only_location=False, n_steps=1):
    """
    Give the input matrix for the regression
    Select only columns in metadata that are features (i.e. not sample_id and firing rates)
    For each input point, concatenate the last n_steps to create one sample.
    """
    X_metadata = []
    firing_rates = []
    i = 0
    while i in range(len(neuron_metadata)):
        if (i - n_steps < 0) or (neuron_metadata[i][0][0] != neuron_metadata[i-n_steps][0][0]):
            # Not enough previous steps available (in the same session)
            i += 1
            continue
        # Remove sampled_id and activations
        row = neuron_metadata[i][factors_start_index:]
        if only_location:
            row = row[:9]
        if add_multiplicative_factors:
            # Add all multiplicative combinations for each factors (e.g. f1*f2, f1*f3, ..., f2*f3, ...)
            from itertools import combinations
            all_factor_index = np.arange(len(row))
            for (f1, f2) in combinations(all_factor_index, 2):
                row.append(row[f1]*row[f2])
        for step in range(1, n_steps):
            # concatenate n-1 previous steps to the sample
            row.extend(neuron_metadata[i-step][factors_start_index:])
        X_metadata.append(row)
        firing_rates.append(neuron_metadata[i][firing_index])
        if delete_n_steps_overlap:
            i = i + n_steps
        else:
            i += 1
    X_metadata = np.array(X_metadata)
    firing_rates = np.array(firing_rates)
    return X_metadata, firing_rates


def timeseries_crosval_predict(X, y, reg_model, nb_folds=10, pca_dim=None):
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
        train_X = np.array(X)[train_ind, :]
        test_X = np.array(X)[test_ind, :]
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim)
            pca.fit(train_X)
            train_X = pca.transform(train_X)
            test_X = pca.transform(test_X)
            # print(sum(pca.explained_variance_ratio_))
        reg_model.fit(train_X, np.array(y)[train_ind])
        if isinstance(reg_model, Pipeline):
            kfold_weights.append(reg_model['reg'].coef_)
        else:
            kfold_weights.append(reg_model.coef_)
        shuffled_predicted_ff = reg_model.predict(test_X)
        for j in range(len(test_ind)):
            predicted_ff[test_ind[j]] = shuffled_predicted_ff[j]
    return predicted_ff, reg_model, kfold_weights


def plot_scatter(title, neuron_id, predicted_ff, firing_rates, save_fig):
    plt.scatter(firing_rates, predicted_ff)
    plt.title(f"{neuron_id[0] + neuron_id[5:].replace('_', '.')} {title}")
    fig = plt.gcf()
    max_ff = np.maximum(np.max(firing_rates), np.max(firing_rates))
    plt.plot([0, max_ff], [0, max_ff], '--', alpha=0.5, color='black')
    without_nan = (np.isnan(firing_rates) + np.isnan(predicted_ff)) == False
    corr, pvalue = pearsonr(np.array(firing_rates)[without_nan], np.array(predicted_ff)[without_nan])
    plt.annotate("$\it{r}$"+f" = {corr:.2f}", (0.01, 0.94), xycoords='axes fraction')
    plt.xlabel(f'Actual firing rates (Hz)')
    plt.ylabel(f'Predicted firing rates (Hz)')
    sns.despine()
    plt.tight_layout()
    if save_fig:
        # fig_path = f"/Users/maxim/Documents/phd/Cosyne 2024/images/fig4_metric_example_{neuron_id}.png"
        save_path = f"{save_fig_folder}/{experiment_name}/{title}scatter-predictions.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=900)
        plt.savefig(save_path[:-3]+'svg', bbox_inches='tight', dpi=900, format='svg')
        plt.clf()
        plt.close()
    else:
        plt.show()


"""
Compute score
"""
sns.set_theme(context="talk", style='white', font_scale=1)

reg = reg_type()
normalize_input = 'reg' in list(hyperparam.keys())[0]
if not normalize_input:
    reg_model = reg
else:
    reg_model = Pipeline([('scaler', StandardScaler()), ('reg', reg)])

os.makedirs(f"{folder_path}/reg_output/", exist_ok=True)
if checkpoint_id is None:
    output_file = open(f"{folder_path}/reg_output/{args.model}-{experiment_name}.txt", 'w')
else:
    os.makedirs(f"{folder_path}/reg_output/{args.model}-{experiment_name}/", exist_ok=True)
    output_file = open(f"{folder_path}/reg_output/{args.model}-{experiment_name}/{checkpoint_id}.txt", 'w')
for neuron_id in metadata.keys():
    start = time.time()

    neuron_metadata, X = align_metadata_and_activation(metadata, activations, neuron_id)
    firing_rates = [e[fr_index] for e in neuron_metadata]
    if is_meta_reg:
        X, firing_rates = get_meta_dataset(neuron_metadata, fr_index, factors_start_index,
                                                    add_multiplicative_factors=False,
                                                    delete_n_steps_overlap=False,
                                                    only_location=only_location,
                                                    n_steps=n_steps)

    reg_model.set_params(**hyperparam)
    predicted_ff, reg_model, _ = timeseries_crosval_predict(X, firing_rates, reg_model, nb_folds, pca_dim)
    without_nan = (np.isnan(firing_rates) + np.isnan(predicted_ff)) == False
    corr, pvalue = pearsonr(np.array(firing_rates)[without_nan], np.array(predicted_ff)[without_nan])

    end = time.time()
    print(neuron_id, reg_name, f"Corr={corr}", f"pvalue={pvalue}", end - start)
    output_file.write(f"Neuron: {neuron_id}, Regressor: {reg_name}, Best_result: , Param: {hyperparam}, Runtime: {end - start} seconds ")
    output_file.write(f"R2: , MSE: , Pearsonr: {corr}, Spearmanr: ")
    output_file.write("\n")
    output_file.flush()

    if plot_fig:
        plot_scatter(f"{neuron_id}", neuron_id, predicted_ff, firing_rates, save_fig)

output_file.close()
print(f'Done')
