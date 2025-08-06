import argparse
import time
import torch
import os
from copy import deepcopy
import numpy as np
import h5py
import pathlib
import matplotlib.pyplot as plt
import copy
from gym_minigrid.envs.associativememory import *

from torch_ac.utils.penv import ParallelEnv
import utils
from utils import device


"""
If want activations for synthetic trials:
--env MiniGrid-Associative-MemoryS7R31T200-v0

If want activations for monkey trials:
--env MiniGrid-Associative-MemoryS7RMTM-v0
"""


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model-type", default=None,
                    help="type of model to use (default from model.py)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 1)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--rnn-memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--transition-as-input", action="store_true", default=False,
                    help="Give previous reward and previous action as input to the model. The environment must support reward and action as observation.")
parser.add_argument("--checkpoint", default=None,
                    help="Checkpoint id if want to select a specific ones. If none, select the final checkpoint")
parser.add_argument("--linear-rnn", action="store_true", default=False,
                    help="Replace the rnn_memory with a linear version")
args = parser.parse_args()

# Visualize meta-info and agent observation to help debugging
is_visualizing = False
checkpoint_id = args.checkpoint if args.checkpoint is None else int(args.checkpoint)


def visualize_env(full_image, session_id, trial_id, direction, agent_dir, agent_location, context_color, goal_up_color, goal_up_location, goal_down_color, goal_down_location, object_selected, choice_optimality, rewards, next_action, env, obs_image=None):
    if obs_image is None:
        plt.imshow(full_image[0])
    else:
        agent_obs = env.get_obs_render(obs_image, tile_size=32)
        plt.imshow(agent_obs)
    plt.title(
        f"Sess: {session_id[0]}, Trial: {trial_id[0]}, Dir: {direction[0]}, Head dir: {agent_dir[0]}\n"
        f"Loc: {agent_location[0]}, Context: {context_color[0]}, Goal up: {goal_up_color[0]} {goal_up_location[0]}, Goal down: {goal_down_color[0]} {goal_down_location[0]}\n"
        f"Object selected: {object_selected[0]}, Optimality: {choice_optimality[0]}, Reward: {rewards[0]}, Next action: {next_action}")
    plt.tight_layout()
    plt.show(bbox_inches='tight')

################################ HOOK
def get_activation(name, activation):
    # print(name, activation)
    def hook(model, input, output):
        # print(name)
        if name == 'memory_rnn':
            activation[name + '_hidden_state'] = output[0].detach().cpu().numpy()
            activation[name + '_cell_state'] = output[1].detach().cpu().numpy()
        elif name == 'max':
            activation[name] = output.detach().cpu().numpy()
        elif name == '':
            activation['actions_probability'] = output[0].probs.detach().cpu().numpy()
            activation['value'] = output[1].detach().cpu().numpy()
            activation['final_memory'] = output[2].detach().cpu().numpy()
        elif name == 'attention':
            activation['attn_output'] = output[0].detach().cpu().numpy()
            activation['attn_output_weights'] = output[1].detach().cpu().numpy() # averaged across heads
        elif '.' in name: # Do not keep intermediary layer activation to save memory space
            pass
        else:
            activation[name] = output.detach().cpu().numpy()
    return hook
################################ HOOK END

def get_session_metadata(metadata, session_id):
    sess = session_id[0].split('\\')[0][0] + session_id[0].split('\\')[1]  # format session_id to match metadata formating
    neuron_id = [e for e in list(metadata.keys()) if sess in e][0]  # select a neuron that was recorded during the current session (doesn't matter which, it only change the firing rates which is not used)
    session_metadata = metadata[neuron_id]

    # Remove irregularities
    samples_to_remove = []
    for i in range(len(session_metadata)-2):
        loc_i = session_metadata[i][0][2]

        # (e.g. monkey almost select a goal but change its mind,
        # i.e. goes: loc 3 -> loc 2 (without reaching the goal) -> loc 1)
        if (loc_i == 3 and session_metadata[i + 1][0][2] == 2 and session_metadata[i + 2][0][2] == 1) or \
                (loc_i == 7 and session_metadata[i + 1][0][2] == 8 and session_metadata[i + 2][0][2] == 9):

            # Make sure that [i+1] or [i+2] are indeed the last location in the trial and that i+3 is another trial.
            if session_metadata[i + 3][0][1] != session_metadata[i][0][1]:
                # remove the end-arm location that is not the final location, i.e. that is not the starting location of the next trial
                next_trial_starting_loc = session_metadata[i + 3][0][2]
                if next_trial_starting_loc != session_metadata[i + 1][0][2]:
                    samples_to_remove.append(session_metadata[i + 1][0])
                else:
                    samples_to_remove.append(session_metadata[i + 2][0])

    session_metadata = [e for e in session_metadata if not e[0] in samples_to_remove]
    return session_metadata

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
utils.seed(args.seed)

print(f"Device: {device}\n")

########## Load environments #########
envs = []
for i in range(args.procs):
    env = utils.make_env(args.env, args.seed + 10000 * i)
    envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")

########## Load agent #########
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, num_envs=args.procs,
                    use_rnn_memory=args.rnn_memory, use_text=args.text, use_transition_as_input=args.transition_as_input,
                    model_type=args.model_type, nb_tasks=envs[0].changing_reward, checkpoint_id=checkpoint_id, linear_rnn=args.linear_rnn)
print("Agent loaded\n")

activation = {}
for name, module in agent.acmodel.named_modules():
    module.register_forward_hook(get_activation(name, activation))


logs = {"num_frames_per_episode": [], "return_per_episode": []}

########## Get activations #########

# if checkpoint, path to checkpoint folder
if checkpoint_id is not None:
    h5_file_path = f'activations_{args.model}_{args.env}/activations_{args.model}_{args.env}_{checkpoint_id}.h5'
else:
    h5_file_path = f'activations_{args.model}_{args.env}.h5'

# Check if local or on CC
os.makedirs(f'activations_{args.model}_{args.env}/', exist_ok=True)

# Check if hdf5 exists
if os.path.isfile(h5_file_path):
    print('Activation hdf5 already exists')
else:
    print('creating hdf5 file')
    ########## Run agent ##########
    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    with h5py.File(h5_file_path, 'a') as hf:
        activations = hf.require_group('layer_activations')
        env_data = hf.require_group('env_data')
        dt = h5py.string_dtype(encoding='utf-8')
        while log_done_counter < args.episodes:
            actions = agent.get_actions(obss)

            #Get activation, observation, previous rewards, current action, dones, agent location in the maze, and image of the full maze
            if len(activations) == 0:
                for k in activation.keys():
                    activations.create_dataset(k, data=activation[k], compression="gzip", chunks=True, maxshape=(None, *activation[k].shape[1:]))
                if len(obss) > 1:
                    # Assume observation from one env. Parallel envs is not implemented
                    raise NotImplementedError
                for obs in obss:
                    for k in obs.keys(): # dict 'image' (agent observation), 'direction', 'mission'
                        if k == 'mission':
                            continue
                        if not isinstance(obs[k], np.ndarray):
                            obs_k = np.array([obs[k]])
                        else:
                            obs_k = obs[k]
                        env_data.create_dataset(f'observation_{k}', data=obs_k, compression="gzip", chunks=True, maxshape=(None, *obs_k.shape[1:]))
                env_data.create_dataset('prev_reward', data=[0.0], compression="gzip", chunks=True, maxshape=(None,)) # TODO  dtype=float?
                env_data.create_dataset('action', data=actions, compression="gzip", chunks=True, maxshape=(None,))
                env_data.create_dataset('done', data=[False], compression="gzip", chunks=True, maxshape=(None,))
                full_image = np.array([env.envs[0].render(mode=None)])
                env_data.create_dataset('full_image', data=full_image, compression="gzip", chunks=True, maxshape=(None, *full_image.shape[1:]))
                agent_location = np.array([env.envs[0].agent_pos])
                env_data.create_dataset('agent_location', data=agent_location, compression="gzip", chunks=True, maxshape=(None, *agent_location.shape[1:]))
                # head direction: agent_dir
                agent_dir = np.array([env.envs[0].agent_dir])
                env_data.create_dataset('head_direction', data=agent_dir, compression="gzip", chunks=True, maxshape=(None, *agent_dir.shape[1:]))
                dt = h5py.string_dtype(encoding='utf-8')
                if 'RMTM' in args.env:
                    trial_id = np.array([env.envs[0].monkey_data[env.envs[0].index_preset]['trial_ids'][env.envs[0].trial_count]], dtype=dt)
                    env_data.create_dataset('trial_id', data=trial_id, compression="gzip", chunks=True, maxshape=(None, *trial_id.shape[1:]))
                    session_id = np.array([env.envs[0].monkey_data[env.envs[0].index_preset]['session_name']], dtype=dt)
                    env_data.create_dataset('session_id', data=session_id, compression="gzip", chunks=True, maxshape=(None, *session_id.shape[1:]))
                else:
                    # use the trial_count as trial_id
                    trial_id = np.array([str(env.envs[0].trial_count)], dtype=dt)
                    env_data.create_dataset('trial_id', data=trial_id, compression="gzip", chunks=True, maxshape=(None, *trial_id.shape[1:]))
                    # session ID
                    session_id = np.array([str(env.envs[0].index_preset) + str(log_done_counter)], dtype=dt)
                    env_data.create_dataset('session_id', data=session_id, compression="gzip", chunks=True, maxshape=(None, *session_id.shape[1:]))
                if 'Associative-Memory' in args.env:
                    purple_context_reward_values_worst_to_best = np.expand_dims(np.array(list(env.envs[0].reward_dict['purple'].keys()), dtype=dt), axis=0)
                    env_data.create_dataset('purple_context_reward_values_worst_to_best', data=purple_context_reward_values_worst_to_best, compression="gzip", chunks=True, maxshape=(None, *purple_context_reward_values_worst_to_best.shape[1:]))
                    # context
                    is_context_still_visible = False
                    context_obj = env.envs[0].cue_obj
                    if not context_obj.cur_pos is None: # if context is visible
                        context_color = np.array([context_obj.color], dtype=dt)
                    else:
                        context_color = np.array([str(None)], dtype=dt)
                    env_data.create_dataset('context', data=context_color, compression="gzip", chunks=True, maxshape=(None, *context_color.shape[1:]))
                    # goals
                    if not env.envs[0].goal_obj_up.cur_pos is None: # if goals are visible
                        goal_up_color = np.array([env.envs[0].goal_obj_up.color], dtype=dt)
                        goal_up_location = np.array([env.envs[0].goal_obj_up.cur_pos])
                        goal_down_color = np.array([env.envs[0].goal_obj_down.color], dtype=dt)
                        goal_down_location = np.array([env.envs[0].goal_obj_down.cur_pos])
                    else:
                        goal_up_color = np.array([str(None)], dtype=dt)
                        goal_up_location = np.array([[-1, -1]])
                        goal_down_color = np.array([str(None)], dtype=dt)
                        goal_down_location = np.array([[-1, -1]])
                    env_data.create_dataset('goal_up', data=goal_up_color, compression="gzip", chunks=True, maxshape=(None, *goal_up_color.shape[1:]))
                    env_data.create_dataset('goal_up_location', data=goal_up_location, compression="gzip", chunks=True, maxshape=(None, *goal_up_location.shape[1:]))
                    env_data.create_dataset('goal_down', data=goal_down_color, compression="gzip", chunks=True, maxshape=(None, *goal_down_color.shape[1:]))
                    env_data.create_dataset('goal_down_location', data=goal_down_location, compression="gzip", chunks=True, maxshape=(None, *goal_down_location.shape[1:]))

                    # Object picked, choice_optimality,
                    #  reward
                    object_selected = np.array([str(None)], dtype=dt)
                    choice_optimality = np.array([str(None)], dtype=dt)
                    if env.envs[0].goal_obj_up.cur_pos is not None: # if goals are visible, check if selecting one of them
                        if tuple(env.envs[0].agent_pos) == (env.envs[0].goal_obj_up.cur_pos[0], env.envs[0].goal_obj_up.cur_pos[1]):
                            object_selected = np.array([env.envs[0].goal_obj_up.color], dtype=dt)
                            reward = env.envs[0].reward_dict[env.envs[0].cue_obj.color][env.envs[0].goal_obj_up.color]
                            choice_optimality = np.array([reward > env.envs[0].reward_dict[env.envs[0].cue_obj.color][env.envs[0].goal_obj_down.color]], dtype=dt)
                        elif tuple(env.envs[0].agent_pos) == (env.envs[0].goal_obj_down.cur_pos[0], env.envs[0].goal_obj_down.cur_pos[1]): # made decision to go to the down object and end trial
                            object_selected = np.array([env.envs[0].goal_obj_down.color], dtype=dt)
                            reward = env.envs[0].reward_dict[env.envs[0].cue_obj.color][env.envs[0].goal_obj_down.color]
                            choice_optimality = np.array([reward > env.envs[0].reward_dict[env.envs[0].cue_obj.color][env.envs[0].goal_obj_up.color]], dtype=dt)
                    env_data.create_dataset('object_chosen', data=object_selected, compression="gzip", chunks=True, maxshape=(None, *object_selected.shape[1:]))
                    env_data.create_dataset('choice_optimality', data=choice_optimality, compression="gzip", chunks=True, maxshape=(None, *choice_optimality.shape[1:]))

                    # direction:
                    if env.envs[0].hallway_start > env.envs[0].hallway_end: # going from right to left
                        direction = np.array(['left'], dtype=dt)
                    else:
                        direction = np.array(['right'], dtype=dt)
                    env_data.create_dataset('direction', data=direction, compression="gzip", chunks=True, maxshape=(None, *direction.shape[1:]))

            else:
                for k in activation.keys():
                    hf[f'layer_activations/{k}'].resize((hf[f'layer_activations/{k}'].shape[0] + activation[k].shape[0]), axis=0)
                    hf[f'layer_activations/{k}'][-activation[k].shape[0]:] = activation[k]
                if len(actions) == 1: # if one process / not using parallel environments
                    for obs in obss:
                        for k in obs.keys():
                            if k == 'mission':
                                continue
                            if not isinstance(obs[k], np.ndarray):
                                obs_k = np.array([obs[k]])
                            else:
                                obs_k = obs[k]
                            hf[f'env_data/observation_{k}'].resize((hf[f'env_data/observation_{k}'].shape[0] + obs_k.shape[0]), axis=0)
                            hf[f'env_data/observation_{k}'][-obs_k.shape[0]:] = obs_k
                    hf['env_data/prev_reward'].resize((hf['env_data/prev_reward'].shape[0] + np.array(rewards).shape[0]), axis=0)
                    hf['env_data/prev_reward'][-np.array(rewards).shape[0]:] = rewards
                    hf['env_data/action'].resize((hf['env_data/action'].shape[0] + actions.shape[0]), axis=0)
                    hf['env_data/action'][-actions.shape[0]:] = actions
                    hf['env_data/done'].resize((hf['env_data/done'].shape[0] + np.array(dones).shape[0]), axis=0)
                    hf['env_data/done'][-np.array(dones).shape[0]:] = dones
                    full_image = np.array([env.envs[0].render(mode=None)])
                    hf['env_data/full_image'].resize((hf['env_data/full_image'].shape[0] + full_image.shape[0]), axis=0)
                    hf['env_data/full_image'][-full_image.shape[0]:] = full_image
                    agent_location = np.array([env.envs[0].agent_pos])
                    hf['env_data/agent_location'].resize((hf['env_data/agent_location'].shape[0] + agent_location.shape[0]), axis=0)
                    hf['env_data/agent_location'][-agent_location.shape[0]:] = agent_location
                    agent_dir = np.array([env.envs[0].agent_dir])
                    hf['env_data/head_direction'].resize((hf['env_data/head_direction'].shape[0] + agent_dir.shape[0]), axis=0)
                    hf['env_data/head_direction'][-agent_dir.shape[0]:] = agent_dir
                    if 'RMTM' in args.env:
                        trial_id = np.array([env.envs[0].monkey_data[env.envs[0].index_preset]['trial_ids'][env.envs[0].trial_count]], dtype=dt)
                        session_id = np.array([env.envs[0].monkey_data[env.envs[0].index_preset]['session_name']], dtype=dt)
                    else:
                        trial_id = np.array([str(env.envs[0].trial_count)], dtype=dt)
                        session_id = np.array([str(env.envs[0].index_preset) + str(log_done_counter)], dtype=dt)
                    hf['env_data/session_id'].resize((hf['env_data/session_id'].shape[0] + session_id.shape[0]), axis=0)
                    hf['env_data/session_id'][-session_id.shape[0]:] = session_id

                    if env.envs[0].hallway_start > env.envs[0].hallway_end:  # going from right to left
                        direction = np.array(['left'], dtype=dt)
                    else:
                        direction = np.array(['right'], dtype=dt)

                    if is_extending_trial_step:
                        # Fix that the trial changes one step to early compared to the monkey env.
                        # consider one additional step in the current trial before changing to the next one
                        hf['env_data/direction'].resize((hf['env_data/direction'].shape[0] + previous_direction.shape[0]), axis=0)
                        hf['env_data/direction'][-previous_direction.shape[0]:] = previous_direction
                        hf['env_data/trial_id'].resize((hf['env_data/trial_id'].shape[0] + previous_trial_id.shape[0]), axis=0)
                        hf['env_data/trial_id'][-previous_trial_id.shape[0]:] = previous_trial_id
                    else:
                        hf['env_data/direction'].resize((hf['env_data/direction'].shape[0] + direction.shape[0]), axis=0)
                        hf['env_data/direction'][-direction.shape[0]:] = direction
                        hf['env_data/trial_id'].resize((hf['env_data/trial_id'].shape[0] + trial_id.shape[0]), axis=0)
                        hf['env_data/trial_id'][-trial_id.shape[0]:] = trial_id

                    if 'Associative-Memory' in args.env:
                        purple_context_reward_values_worst_to_best = np.expand_dims(np.array(list(env.envs[0].reward_dict['purple'].keys()), dtype=dt), axis=0)
                        hf['env_data/purple_context_reward_values_worst_to_best'].resize((hf['env_data/purple_context_reward_values_worst_to_best'].shape[0] + purple_context_reward_values_worst_to_best.shape[0]), axis=0)
                        hf['env_data/purple_context_reward_values_worst_to_best'][-purple_context_reward_values_worst_to_best.shape[0]:] = purple_context_reward_values_worst_to_best
                        if not previous_context.cur_pos is None or is_context_still_visible:
                            # if context appeared and didn't disappear yet
                            is_context_still_visible = True
                            previous_context_color = np.array([previous_context.color], dtype=dt)  # Agent sees the context one step later than when it is added to the env
                        else:
                            previous_context_color = np.array([str(None)], dtype=dt)
                        hf['env_data/context'].resize((hf['env_data/context'].shape[0] + previous_context_color.shape[0]), axis=0)
                        hf['env_data/context'][-previous_context_color.shape[0]:] = previous_context_color
                        if not previous_goal_up.cur_pos is None:  # if goals are visible
                            goal_up_color = np.array([previous_goal_up.color], dtype=dt)
                            goal_up_location = np.array([previous_goal_up.cur_pos])
                            goal_down_color = np.array([previous_goal_down.color], dtype=dt)
                            goal_down_location = np.array([previous_goal_down.cur_pos])
                        else:
                            goal_up_color = np.array([str(None)], dtype=dt)
                            goal_up_location = np.array([[-1, -1]])
                            goal_down_color = np.array([str(None)], dtype=dt)
                            goal_down_location = np.array([[-1, -1]])
                        hf['env_data/goal_up'].resize((hf['env_data/goal_up'].shape[0] + goal_up_color.shape[0]), axis=0)
                        hf['env_data/goal_up'][-goal_up_color.shape[0]:] = goal_up_color
                        hf['env_data/goal_up_location'].resize((hf['env_data/goal_up_location'].shape[0] + goal_up_location.shape[0]), axis=0)
                        hf['env_data/goal_up_location'][-goal_up_location.shape[0]:] = goal_up_location
                        hf['env_data/goal_down'].resize((hf['env_data/goal_down'].shape[0] + goal_down_color.shape[0]), axis=0)
                        hf['env_data/goal_down'][-goal_down_color.shape[0]:] = goal_down_color
                        hf['env_data/goal_down_location'].resize((hf['env_data/goal_down_location'].shape[0] + goal_down_location.shape[0]), axis=0)
                        hf['env_data/goal_down_location'][-goal_down_location.shape[0]:] = goal_down_location

                        object_selected = np.array([str(None)], dtype=dt)
                        choice_optimality = np.array([str(None)], dtype=dt)
                        if previous_goal_up.cur_pos is not None: # env.envs[0].goal_obj_up.cur_pos is not None:  # if goals are visible, check if selecting one of them
                            if tuple(env.envs[0].agent_pos) == previous_goal_up.cur_pos: # (env.envs[0].goal_obj_up.cur_pos[0], env.envs[0].goal_obj_up.cur_pos[1]):
                                object_selected = np.array([previous_goal_up.color], dtype=dt) # np.array([env.envs[0].goal_obj_up.color], dtype=dt)
                                reward = env.envs[0].reward_dict[previous_context.color][previous_goal_up.color] # env.envs[0].reward_dict[env.envs[0].cue_obj.color][env.envs[0].goal_obj_up.color]
                                choice_optimality = np.array([str(reward > env.envs[0].reward_dict[previous_context.color][previous_goal_down.color])], dtype=dt)
                            elif tuple(env.envs[0].agent_pos) == previous_goal_down.cur_pos: # (env.envs[0].goal_obj_down.cur_pos[0], env.envs[0].goal_obj_down.cur_pos[1]):  # made decision to go to the down object and end trial
                                object_selected = np.array([previous_goal_down.color], dtype=dt)# np.array([env.envs[0].goal_obj_down.color], dtype=dt)
                                reward = env.envs[0].reward_dict[previous_context.color][previous_goal_down.color]# env.envs[0].reward_dict[env.envs[0].cue_obj.color][env.envs[0].goal_obj_down.color]
                                choice_optimality = np.array([str(reward > env.envs[0].reward_dict[previous_context.color][previous_goal_up.color])], dtype=dt)# np.array([reward > env.envs[0].reward_dict[env.envs[0].cue_obj.color][env.envs[0].goal_obj_up.color]], dtype=dt)
                        hf['env_data/object_chosen'].resize((hf['env_data/object_chosen'].shape[0] + object_selected.shape[0]), axis=0)
                        hf['env_data/object_chosen'][-object_selected.shape[0]:] = object_selected
                        hf['env_data/choice_optimality'].resize((hf['env_data/choice_optimality'].shape[0] + choice_optimality.shape[0]), axis=0)
                        hf['env_data/choice_optimality'][-choice_optimality.shape[0]:] = choice_optimality
                        if previous_trial_id != trial_id: # context disappear when the trial is over
                            is_context_still_visible = False
                    if is_visualizing:
                        if is_extending_trial_step:
                            visualize_env(full_image, session_id, previous_trial_id, previous_direction, agent_dir, agent_location, previous_context_color,
                                          goal_up_color, goal_up_location, goal_down_color, goal_down_location, object_selected,
                                          choice_optimality, rewards, actions[0], env.envs[0], obss[0]['image'])
                        else:
                            visualize_env(full_image, session_id, trial_id, direction, agent_dir, agent_location, previous_context_color,
                                          goal_up_color, goal_up_location, goal_down_color, goal_down_location, object_selected,
                                          choice_optimality, rewards, actions[0], env.envs[0], obss[0]['image'])
                else:
                    # if want to parallelize envs, need to look if it correctly aligns activation and data between parallel envs
                    # Also, need to change when adding first item in the dictionary, e.g. actions[0]
                    raise NotImplementedError

            # Goals and context appears and disappear one step early in env compared to what the agent sees
            previous_goal_up = copy.deepcopy(env.envs[0].goal_obj_up)
            previous_goal_down = copy.deepcopy(env.envs[0].goal_obj_down)
            previous_context = copy.deepcopy(env.envs[0].cue_obj)
            # Fix the problem of having no step in the end-arm. Count one extra step in the current trial before switching to the next
            previous_session_id = copy.deepcopy(session_id)
            previous_trial_id = copy.deepcopy(trial_id)
            previous_direction = copy.deepcopy(direction)
            previous_agent_pos = copy.deepcopy(env.envs[0].agent_pos)
            previous_trial_count = env.envs[0].trial_count

            obss, rewards, dones, _ = env.step(actions)
            agent.analyze_feedbacks(rewards, dones)


            # Check if change session
            if 'RMTM' in args.env:
                trial_id = np.array([env.envs[0].monkey_data[env.envs[0].index_preset]['trial_ids'][env.envs[0].trial_count]], dtype=dt)
                session_id = np.array([env.envs[0].monkey_data[env.envs[0].index_preset]['session_name']], dtype=dt)
            else:
                trial_id = np.array([str(env.envs[0].trial_count)], dtype=dt)
                session_id = np.array([str(env.envs[0].index_preset) + str(log_done_counter)], dtype=dt)

            if previous_session_id != session_id:
                # If change session, do not use 'previous' variables from another session
                previous_goal_up = copy.deepcopy(env.envs[0].goal_obj_up)
                previous_goal_down = copy.deepcopy(env.envs[0].goal_obj_down)
                previous_context = copy.deepcopy(env.envs[0].cue_obj)
                previous_session_id = copy.deepcopy(session_id)
                previous_trial_id = copy.deepcopy(trial_id)
                if env.envs[0].hallway_start > env.envs[0].hallway_end:  # going from right to left
                    direction = np.array(['left'], dtype=dt)
                else:
                    direction = np.array(['right'], dtype=dt)
                previous_direction = copy.deepcopy(direction)
                # Also doesn't want the reward from the previous session
                rewards = (0.0,)

            if previous_trial_id != trial_id and previous_session_id == session_id \
                    and previous_agent_pos[0] == env.envs[0].agent_pos[0] and abs(previous_agent_pos[1] - env.envs[0].agent_pos[1]) == 1:
                    is_extending_trial_step = True
            else:
                is_extending_trial_step = False

            log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
            log_episode_num_frames += torch.ones(args.procs, device=device)

            for i, done in enumerate(dones):
                if done:
                    hf.flush()
                    print(f'flushed episode {env.envs[0].index_preset} ({log_done_counter+1} episode dones)')
                    log_done_counter += 1
                    logs["return_per_episode"].append(log_episode_return[i].item())
                    logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

            mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
            log_episode_return *= mask
            log_episode_num_frames *= mask

    end_time = time.time()
    print(f'hdf5 completed in {end_time-start_time} seconds')

