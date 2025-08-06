import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

save_fig_folder = f"rl/figures/neurodata/"

neuron_color = '#4F5669'

pickle_in = open('rl/metadata.pickle', "rb")
metadata = pickle.load(pickle_in)
pickle_in.close()
sample_index = 0
fr_index = 1
factors_start_index = 2

num_trials_per_neuron = []
for n in metadata.keys():
    trial_ids = [e[1] for e in np.array(metadata[n])[:, sample_index]]
    num_trials_per_neuron.append(len(set(trial_ids)))


sns.set_theme(context="talk", style='white', font_scale=1.5)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 15))

# Histogram with KDE overlay
sns.histplot(num_trials_per_neuron, bins=20, kde=True, color=neuron_color, edgecolor='black', alpha=0.7, ax=ax)

# Labels and title
ax.set_xlabel("Number of trials per neuron")
ax.set_ylabel("Count")
sns.despine()

os.makedirs(f"{save_fig_folder}/svg/", exist_ok=True)
plt.savefig(f"{save_fig_folder}/svg/num_trials_per_neuron.svg", bbox_inches='tight', format='svg')
plt.close()
plt.clf()

# Show plot
plt.show()




print('.')
