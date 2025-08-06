from glob import glob
import scipy.io as sio
import numpy
import pickle
# Python code to extract from the xmaze.mat the ranking of the colors for each sessions:
root_folder = "E:\Bashivan Lab\monkey-data-j\Results\\"
monkey_folder = ['Raul\\', 'Woody\\']
sessions = glob(root_folder + monkey_folder[0] + "*\*_XMaze.mat", recursive=True)
raul_session_index = (0, len(sessions))
sessions.extend(glob(root_folder + monkey_folder[1] + "*\*_XMaze.mat", recursive=True))
sessions.sort()
# Extract the reward setting of each session
reward_settings = []
sess = []
outcome_per_session = []
colors_per_session = []
context_per_session = []
for s in sessions:
    mat_contents = sio.loadmat(s, squeeze_me=True, struct_as_record=False)
    sess.append('\\'.join(s.split('\\')[-3:-1]))
    reward_settings.append(mat_contents['XMazeStruct'].SessionInfo.Context1NParams[1].tolist())
    outcome_per_session.append(mat_contents['XMazeStruct'].TrialSummary[1:, 6].tolist()) # Get outcome = 'Correct' or 'Incorrect' for each trial in a session
    colors_per_session.append(mat_contents['XMazeStruct'].TrialSummary[1:, 4:6].tolist())
    context_per_session.append(mat_contents['XMazeStruct'].TrialSummary[1:, 3].tolist())

pickle_out = open("monkeyColors","wb")
pickle.dump(colors_per_session, pickle_out)
pickle_out.close()

pickle_out = open("monkeyContext","wb")
pickle.dump(context_per_session, pickle_out)
pickle_out.close()

pickle_out = open("monkeyOutcomes","wb")
pickle.dump(outcome_per_session, pickle_out)
pickle_out.close()

numpy.save('rewardSettings.npy', reward_settings, allow_pickle=True)
numpy.save('monkeySessions.npy', sess, allow_pickle=True)
