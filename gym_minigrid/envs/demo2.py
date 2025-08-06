import numpy as np
import pickle
import os
from pathlib import Path

# Read monkey experiment info from .npy and pickle files
monkey_sessions = np.load('E:\BashivanLab\Code\\2d-memory\gym_minigrid\envs\\monkeySessions.npy')

monkey_reward_settings = np.load('E:\BashivanLab\Code\\2d-memory\gym_minigrid\envs\\rewardSettings.npy')
# monkey_reward_settings = np.load('/Users/maxim/PycharmProjects/2d-memory/gym_minigrid/envs/rewardSettings.npy')
monkey_reward_settings = np.char.lower(monkey_reward_settings)
monkey_reward_settings = np.char.replace(monkey_reward_settings, 'cyan', 'grey')

pickle_in = open("E:\BashivanLab\Code\\2d-memory\gym_minigrid\envs\\monkeyOutcomes","rb")
# pickle_in = open("/Users/maxim/PycharmProjects/2d-memory/gym_minigrid/envs/monkeyOutcomes","rb")
monkey_outcomes = pickle.load(pickle_in)
pickle_in.close()
del monkey_outcomes[6][0]
del monkey_outcomes[6][0]
del monkey_outcomes[3][0]
invalid_index = []
for i, x in enumerate(monkey_outcomes):
    if 'userStoppedTrial' in x:
        invalid_index.append((i, x.index('userStoppedTrial')))

pickle_in = open("E:\BashivanLab\Code\\2d-memory\gym_minigrid\envs\\monkeyColors","rb")
# pickle_in = open("/Users/maxim/PycharmProjects/2d-memory/gym_minigrid/envs/monkeyColors","rb")
monkey_colors = pickle.load(pickle_in)
pickle_in.close()
del monkey_colors[6][0]
del monkey_colors[6][0]
del monkey_colors[3][0]
for i in invalid_index:
    x, y = i
    #del monkey_colors[x][y]
    monkey_colors[x][y] = None
for i in range(len(monkey_colors)):
    for j in range(len(monkey_colors[i])):
        if monkey_colors[i][j] is None or len(monkey_colors[i][j][0]) == 0:
            continue
        monkey_colors[i][j] = np.char.lower(monkey_colors[i][j])
        monkey_colors[i][j] = np.char.replace(monkey_colors[i][j], 'cyan', 'grey')   

pickle_in = open("E:\BashivanLab\Code\\2d-memory\gym_minigrid\envs\\monkeyContext","rb")
# pickle_in = open("/Users/maxim/PycharmProjects/2d-memory/gym_minigrid/envs/monkeyContext","rb")
monkey_context = pickle.load(pickle_in)
pickle_in.close()
del monkey_context[6][0]
del monkey_context[6][0]
del monkey_context[3][0]
for i in invalid_index:
    x, y = i
    #del monkey_context[x][y]
    monkey_context[x][y] = None
for i in range(len(monkey_context)):
    for j in range(len(monkey_context[i])):
        if monkey_context[i][j] is None or len(monkey_context[i][j]) == 0:
            continue
        monkey_context[i][j] = np.char.replace(monkey_context[i][j], 'Steel', 'Purple')
        monkey_context[i][j] = np.char.replace(monkey_context[i][j], 'Wood', 'Yellow')
        monkey_context[i][j] = np.char.lower(monkey_context[i][j])

pickle_in = open("E:\BashivanLab\Code\\2d-memory\gym_minigrid\envs\\monkeyTrialIDs","rb")
# pickle_in = open("/Users/maxim/PycharmProjects/2d-memory/gym_minigrid/envs/monkeyContext","rb")
monkey_trials = pickle.load(pickle_in)
pickle_in.close()
del monkey_trials[6][0]
del monkey_trials[6][0]
del monkey_trials[3][0]
for i in invalid_index:
    x, y = i
    #del monkey_trials[x][y]
    monkey_trials[x][y] = None

monkey_data = []
numSessions = len(monkey_sessions)
for i in range(numSessions):
    dict = {}
    dict['session_name'] = monkey_sessions[i]
    dict['trial_ids'] = monkey_trials[i]
    dict['reward_setting'] = monkey_reward_settings[i]
    dict['context_colors'] = monkey_context[i]
    dict['goal_colors'] = monkey_colors[i]
    dict['outcomes'] = monkey_outcomes[i]
    monkey_data.append(dict)

pickle_out = open("../../../../Downloads/monkeyData", "wb")
pickle.dump(monkey_data, pickle_out)
pickle_out.close()