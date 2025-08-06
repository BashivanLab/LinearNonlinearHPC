import random

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
# import itertools
# from random import randrange
# from random import sample
import numpy as np
import pickle
from pathlib import Path

class AssociativeMemoryEnv(MiniGridEnv):
    """
    This environment is a memory test. The agent starts in a small room
    where it sees an object. It then has to go through a narrow hallway
    which ends in a split. At each end of the split there is an object,
    one of which is the same as the object in the starting room. The
    agent has to remember the initial object, and go to the matching
    object at split.
    """
    def __init__(
        self,
        seed,
        size=8,
        random_length=False,
        nb_trial_per_episode=2500, # nb_trial_per_episode=100, # nb_trial_per_episode=1, #
        changing_reward=False,
        side_indicator=False,
        distal_cue=True,
        max_steps=(100*9)**3,
        max_trial_steps=80,
        curriculum_learning=False
    ):
        self.nb_trial_per_episode = nb_trial_per_episode
        self.trial_count = 0
        self.changing_reward = changing_reward # Change reward schema every episode
        self.random_length = random_length
        self.side_indicator = side_indicator
        self.distal_cue = distal_cue
        self.is_curriculum_learning = curriculum_learning
        self.goal_colors = ['red', 'blue', 'green']
        self.available_colors = {'purple', 'yellow', 'grey', 'red', 'blue', 'green', 'orange'}


        if self.is_curriculum_learning:
            # keep track
            self.trial_optimality = []
            if isinstance(self.changing_reward, int):
                # collect the trials optimality for each reward schema
                self.plateau_size = self.changing_reward * self.nb_trial_per_episode
            else:
                print('Curriculum learning not implemented for this reward schemas')
                raise

        if self.changing_reward == 'monkey': # Replicate monkey experiment's reward setting per sessions into the environment with the agent
            # Read monkey experiment data info from monkeyData pickle files
            data_folder = Path(__file__).parent
            file_to_open = data_folder / "monkeyData"
            pickle_in = open(file_to_open, "rb")
            self.monkey_data = pickle.load(pickle_in)
            pickle_in.close()

            # Use preset_goals to determine the task_index
            self.preset_goals = [['orange', 'blue', 'red'], ['yellow', 'blue', 'red'], ['purple', 'blue', 'red'], ['grey', 'blue', 'green'],
                                    ['blue', 'grey', 'purple'], ['yellow', 'red', 'grey'], ['blue', 'orange', 'red'],
                                    ['orange', 'grey', 'green'], ['purple', 'red', 'yellow'], ['purple', 'red', 'blue'],
                                    ['green', 'orange', 'yellow'], ['grey', 'red', 'green'], ['orange', 'yellow', 'purple'],
                                    ['blue', 'green', 'grey'], ['yellow', 'red', 'green'], ['purple', 'orange', 'blue'],
                                    ['orange', 'blue', 'yellow'], ['green', 'grey', 'red'], ['grey', 'green', 'purple'], ['purple', 'yellow', 'green'],
                                    ['grey', 'red', 'blue'], ['purple', 'orange', 'red'], ['yellow', 'blue', 'grey'],
                                    ['red', 'yellow', 'purple'], ['green', 'red', 'yellow'], ['blue', 'grey', 'yellow'],
                                    ['red', 'purple', 'orange'], ['green', 'yellow', 'grey'], ['red', 'grey', 'green'],
                                    ['green', 'purple', 'grey'], ['orange', 'green', 'purple']]
        elif self.changing_reward == 31:
            self.preset_goals = [['orange', 'blue', 'red'], ['yellow', 'blue', 'red'], ['purple', 'blue', 'red'], ['grey', 'blue', 'green'],
                                    ['blue', 'grey', 'purple'], ['yellow', 'red', 'grey'], ['blue', 'orange', 'red'],
                                    ['orange', 'grey', 'green'], ['purple', 'red', 'yellow'], ['purple', 'red', 'blue'],
                                    ['green', 'orange', 'yellow'], ['grey', 'red', 'green'], ['orange', 'yellow', 'purple'],
                                    ['blue', 'green', 'grey'], ['yellow', 'red', 'green'], ['purple', 'orange', 'blue'],
                                    ['orange', 'blue', 'yellow'], ['green', 'grey', 'red'], ['grey', 'green', 'purple'], ['purple', 'yellow', 'green'],
                                    ['grey', 'red', 'blue'], ['purple', 'orange', 'red'], ['yellow', 'blue', 'grey'],
                                    ['red', 'yellow', 'purple'], ['green', 'red', 'yellow'], ['blue', 'grey', 'yellow'],
                                    ['red', 'purple', 'orange'], ['green', 'yellow', 'grey'], ['red', 'grey', 'green'],
                                    ['green', 'purple', 'grey'], ['orange', 'green', 'purple']] # Add missing setting from what is seen in monkeytrials ['green', 'grey', 'red']
        elif self.changing_reward == 30:
            self.preset_goals = [['orange', 'blue', 'red'], ['yellow', 'blue', 'red'], ['purple', 'blue', 'red'],
                                 ['grey', 'blue', 'green'],
                                 ['blue', 'grey', 'purple'], ['yellow', 'red', 'grey'], ['blue', 'orange', 'red'],
                                 ['orange', 'grey', 'green'], ['purple', 'red', 'yellow'], ['purple', 'red', 'blue'],
                                 ['green', 'orange', 'yellow'], ['grey', 'red', 'green'],
                                 ['orange', 'yellow', 'purple'],
                                 ['blue', 'green', 'grey'], ['yellow', 'red', 'green'], ['purple', 'orange', 'blue'],
                                 ['orange', 'blue', 'yellow'], ['grey', 'green', 'purple'], ['purple', 'yellow', 'green'],
                                 ['grey', 'red', 'blue'], ['purple', 'orange', 'red'], ['yellow', 'blue', 'grey'],
                                 ['red', 'yellow', 'purple'], ['green', 'red', 'yellow'], ['blue', 'grey', 'yellow'],
                                 ['red', 'purple', 'orange'], ['green', 'yellow', 'grey'], ['red', 'grey', 'green'],
                                 ['green', 'purple', 'grey'], ['orange', 'green', 'purple']]
        elif self.changing_reward == 18: # 18 fixed reward settings
            self.preset_goals = [['purple', 'yellow', 'grey'], ['red', 'blue', 'green'], ['grey', 'yellow', 'orange'],
                                 ['green', 'blue', 'purple'], ['red', 'grey', 'yellow'], ['green', 'blue', 'purple'],
                                 ['orange', 'yellow', 'red'], ['purple', 'green', 'blue'], ['red', 'yellow', 'orange'],
                                 ['blue', 'grey', 'purple'], ['red', 'orange', 'yellow'], ['purple', 'green', 'blue'],
                                 ['grey', 'yellow', 'red'], ['purple', 'green', 'blue'], ['yellow', 'grey', 'orange'],
                                 ['purple', 'blue', 'green'], ['red', 'grey', 'orange'], ['purple', 'yellow', 'blue']]
        elif self.changing_reward == 8: # 8 fixed reward settings
            self.preset_goals = [['purple', 'yellow', 'grey'], ['red', 'blue', 'green'], ['grey', 'yellow', 'orange'],
                                 ['green', 'blue', 'purple'], ['red', 'grey', 'yellow'], ['green', 'blue', 'purple'],
                                 ['orange', 'yellow', 'red'], ['purple', 'green', 'blue']]
        elif self.changing_reward == 4: # 4 fixed reward settings
            self.preset_goals = [['purple', 'yellow', 'grey'], ['red', 'blue', 'green'], ['grey', 'yellow', 'orange'], ['blue', 'green', 'red']]
        elif self.changing_reward == 2: # Two fixed distributions
            self.preset_goals = [['red', 'blue', 'green'], ['purple', 'yellow', 'grey']]
        else:   # one fixed distribution
            self.preset_goals = [['red', 'blue', 'green']]

        self.index_preset = 0
        self.max_trial_steps = max_trial_steps
        self.trial_step_count = 0

        super().__init__(
            seed=seed,
            grid_size=size,
            max_steps=max_steps, # 5*size**2,
            see_through_walls=True,
            agent_view_size=5,
        )
        # Reduce action space by removing not used actions. Allow only 3 actions permitted: left, right, forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)

    def find_task_index(self, goal_colors):
        # Used for monkey trials
        # find the task_index of the setting with using 'goal_colors' based on the settings in preset_goals
        for i, lst in enumerate(self.preset_goals):
            for j, color in enumerate(lst):
                if color != goal_colors[j]:
                    break
                elif j == 2:
                    return i
        print('Task index not found among the preset goals')
        raise

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Reward values of each goal objects depend on the context / cue object
        best_reward, medium_reward, worst_reward = 1.0, 0.5, 0.1
        reward_values = [best_reward, medium_reward, worst_reward]

        # print(f'######### Switched to reward distribution from {self.goal_colors} ##########')
        if self.changing_reward == True: # every episode, randomlu change the goal colors, without repeating a color from the previous episode
            self.goal_colors = np.random.choice(list(self.available_colors - set(self.goal_colors)), 3, replace=False).flatten().tolist()
            # print(f'######### to {self.goal_colors} ##########')
        elif self.changing_reward == 'monkey':
            self.index_preset = (self.index_preset + 1) % len(self.monkey_data)
            self.trial_count = 0
            self.trial_step_count = 0
            self.goal_colors = self.monkey_data[self.index_preset]['reward_setting']
            self.nb_trial_per_episode = len(self.monkey_data[self.index_preset]['goal_colors'])
            # print(f'######### to {self.goal_colors} ########## preset #: {self.index_preset}')
        else: # fixed number of settings predetermined
            self.index_preset = (self.index_preset + 1) % len(self.preset_goals)
            self.goal_colors = self.preset_goals[self.index_preset]
            # print(f'######### to {self.goal_colors} ########## preset #: {self.index_preset}')

        yellow_rewards = {self.goal_colors[0]: reward_values[0],
                        self.goal_colors[1]: reward_values[1],
                        self.goal_colors[2]: reward_values[2]}
        purple_rewards = {self.goal_colors[0]: reward_values[2],
                        self.goal_colors[1]: reward_values[1],
                        self.goal_colors[2]: reward_values[0]}

        self.reward_dict = {'purple': purple_rewards, 'yellow': yellow_rewards}

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.height = height
        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_entry_right = self._rand_int(8, width - 2)
        else:
            hallway_entry_right = width - 3
        hallway_entry_left = 2
        self.hallway_start = hallway_entry_left
        self.hallway_end = hallway_entry_right

        # Horizontal hallway
        for i in range(hallway_entry_left, hallway_entry_right):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Starting (left) Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_entry_left, j, Wall())
            self.grid.set(hallway_entry_left - 2, j, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_entry_right, j, Wall())
            self.grid.set(hallway_entry_right + 2, j, Wall())

        self.grid.set(hallway_entry_left - 1, self.height // 2 - 2, Wall())
        self.grid.set(hallway_entry_left - 1, self.height // 2 + 2, Wall())
        self.grid.set(hallway_entry_right + 1, self.height // 2 - 2, Wall())
        self.grid.set(hallway_entry_right + 1, self.height // 2 + 2, Wall())

        self.agent_pos = (hallway_entry_left-1, np.random.choice([self.height // 2 + 1, self.height // 2 - 1]))
        if self.agent_pos[1] <= height // 2:
            self.agent_dir = 1
        else:
            self.agent_dir = 3

        # Place objects
        if self.changing_reward == 'monkey':
            self.cue_obj = Square(self.monkey_data[self.index_preset]['context_colors'][self.trial_count].item())
        else:
            self.cue_obj = Square(self._rand_elem(['purple', 'yellow']))

        self.create_goals()

        if self.side_indicator:
            self.add_side_indicator()

        if self.distal_cue:
            self.add_distal_cue()

        self.mission = 'go to the highest value object at the end of the hallway'

    def step(self, action):
        if self.changing_reward == 'monkey':
            if (self.index_preset == 3 and self.trial_count == 0) or (self.index_preset == 3 and self.trial_count == 8) or (self.index_preset == 17 and self.trial_count == 10):
                 print('Skipping because Wrong Color  ', self.trial_count, self.index_preset)
                 self.stop_current_trial_and_start_next_one()

            if (self.monkey_data[self.index_preset]['goal_colors'][self.trial_count] is None) or (
                    self.monkey_data[self.index_preset]['context_colors'][self.trial_count] is None):
                print('Skipping None trial', self.trial_count, self.index_preset)
                self.stop_current_trial_and_start_next_one()
        # if self.index_preset == 17:
        #     print('####### TEMP REMOVE LATER ##### Skipping index_preset 17 because was not trained on it')
        #     self.stop_current_trial_and_start_next_one()

        obs, reward, done, info = MiniGridEnv.step(self, action)
        # reward = -1
        reward = 0.0
        self.trial_step_count += 1

        if self.cue_obj.cur_pos is None: # make the corrridor context cue appears if looks like the agent will enter the corridor
            if self.hallway_start > self.hallway_end:  # going from right to left
                cue_x_offset = -1
                facing_end = 2
            else:
                cue_x_offset = 1
                facing_end = 0
            if (self.agent_pos[0] == self.hallway_start - cue_x_offset) and (self.agent_pos[1] == self.height // 2) and (self.agent_dir == facing_end):
                upper_room_wall = self.height // 2 - 2
                lower_room_wall = self.height // 2 + 2
                self.put_obj(self.cue_obj, self.hallway_start + cue_x_offset, upper_room_wall + 1)
                self.grid.set(self.hallway_start + cue_x_offset, lower_room_wall - 1, self.cue_obj)

        if self.cue_obj.cur_pos is not None: # make the context cue in the corridor disappear after passing the cue, and make those at the end of the maze appear
            start_from_right = self.hallway_start > self.hallway_end
            if (start_from_right and self.agent_pos[0] <= self.cue_obj.cur_pos[0]) or (not start_from_right and self.agent_pos[0] >= self.cue_obj.cur_pos[0]):
                # Remove corridor context cue
                self.grid.set(*self.cue_obj.cur_pos, Wall())
                self.grid.set(self.cue_obj.cur_pos[0], self.cue_obj.cur_pos[1]+2, Wall())
                self.cue_obj.cur_pos = None

                # Add arms context cue
                if self.hallway_start > self.hallway_end:  # going from right to left
                    x_pos_goal = self.hallway_end - 1
                else:
                    x_pos_goal = self.hallway_end + 1
                pos0 = (x_pos_goal, self.height // 2 - 1)
                pos1 = (x_pos_goal, self.height // 2 + 1)
                self.modify_context_cue_next_to_goals(pos0, pos1, 'place')

        if self.goal_obj_up.cur_pos is None and self.agent_pos[0] == self.hallway_end: # if goals are not already placed and reached end of the hallway:
            self.goals_appear()

        if self.goal_obj_up.cur_pos is not None:
            if tuple(self.agent_pos) == (self.goal_obj_up.cur_pos[0], self.goal_obj_up.cur_pos[1]): # made decision to go up and end trial
                reward = self.reward_dict[self.cue_obj.color][self.goal_obj_up.color]
                self.trial_count += 1
                info['is_optimal'] = reward > self.reward_dict[self.cue_obj.color][self.goal_obj_down.color] # if chosed the optimal goal
                info['incomplete_trial'] = False
                if self.trial_count < self.nb_trial_per_episode:
                    self.start_next_trial()
            elif tuple(self.agent_pos) == (self.goal_obj_down.cur_pos[0], self.goal_obj_down.cur_pos[1]): # made decision to go to the down object and end trial
                # if (self.index_preset != 3 and self.trial_count != 0) and (self.index_preset != 17 and self.trial_count != 10):
                reward = self.reward_dict[self.cue_obj.color][self.goal_obj_down.color]
                self.trial_count += 1
                info['is_optimal'] = reward > self.reward_dict[self.cue_obj.color][self.goal_obj_up.color] # if chosed the optimal goal
                info['incomplete_trial'] = False
                if self.trial_count < self.nb_trial_per_episode:
                    self.start_next_trial()

        if self.trial_step_count >= self.max_trial_steps:
            info = self.stop_current_trial_and_start_next_one(info)
            info['incomplete_trial'] = True

        if self.trial_count >= self.nb_trial_per_episode:
            self.trial_count = 0
            self.trial_step_count = 0
            done = True

        if self.is_curriculum_learning:
            """
            Increase the number of trial per episode if, in average across all reward settings, the average optimality is higher than the performance target
            """
            if self.nb_trial_per_episode < 350:

                if 'is_optimal' in info.keys():
                    self.trial_optimality.append(info['is_optimal'])
                    if len(self.trial_optimality) > self.plateau_size:
                        self.trial_optimality = self.trial_optimality[-self.plateau_size:]
                    average_optimality = np.mean(self.trial_optimality)
                    if self.nb_trial_per_episode <= 100:
                        performance_target = 0.97
                    elif self.nb_trial_per_episode <= 200:
                        performance_target = 0.95
                    else:
                        performance_target = 0.95
                    if average_optimality >= performance_target and len(self.trial_optimality) >= self.plateau_size:
                        print(f"###### Number of trial per episode increased from {self.nb_trial_per_episode} to {self.nb_trial_per_episode+10}")
                        self.nb_trial_per_episode += 10
                        self.plateau_size = self.changing_reward * self.nb_trial_per_episode
                        self.trial_optimality = []

        info['nb_trial_per_episode'] = self.nb_trial_per_episode
        obs['reward'] = reward
        obs['action'] = action
        if isinstance(self.changing_reward, int): # if have some preset reward settings
            obs['task_index'] = self.index_preset # give as input the number of the reward setting
        elif self.changing_reward == 'monkey':
            obs['task_index'] = self.find_task_index(self.goal_colors)
        return obs, reward, done, info

    def start_next_trial(self, reverse_direction=True):
        self.trial_step_count = 0
        # Reverse direction in which the agent traverses the maze, i.e. new_hallway_start, new_hallway_end = previous_hallway_end, previous_hallway_start
        # self.hallway_start, hallway_end = self.goal_obj_up.cur_pos[0]-1, self.hallway_start
        if reverse_direction:
            self.hallway_start, self.hallway_end = self.hallway_end, self.hallway_start
        if self.hallway_start > self.hallway_end: # going from right to left
            x_pos_cue = self.hallway_start - 1
        else:
            x_pos_cue = self.hallway_start + 1

        # remove goal and cue objects
        self.modify_context_cue_next_to_goals(self.goal_obj_down.cur_pos, self.goal_obj_up.cur_pos, 'remove')
        self.grid.set(*self.goal_obj_down.cur_pos, None)
        self.grid.set(*self.goal_obj_up.cur_pos, None)

        # Place new goal objects and cue object in the opposite location
        upper_room_wall = self.height // 2 - 2
        lower_room_wall = self.height // 2 + 2

        if self.changing_reward == 'monkey':
            if (self.monkey_data[self.index_preset]['goal_colors'][self.trial_count] is None) or (
                    self.monkey_data[self.index_preset]['context_colors'][self.trial_count] is None) or (
                    'gray' in self.monkey_data[self.index_preset]['goal_colors'][self.trial_count]):
                print('Skipping gray or None', self.trial_count, self.index_preset)
                self.stop_current_trial_and_start_next_one()
            else:
                self.cue_obj = Square(self.monkey_data[self.index_preset]['context_colors'][self.trial_count].item())
                self.create_goals()
        else:
            self.cue_obj = Square(self._rand_elem(['purple', 'yellow']))
            self.create_goals()

    def stop_current_trial_and_start_next_one(self, info=None, starting_location=None):
        reverse_direction = True
        self.trial_step_count = 0
        self.trial_count += 1
        if info is not None:
            info['is_optimal'] = False

        # Set the environment to be ready to start a new trial
        if self.goal_obj_up.cur_pos is None:
            self.goals_appear()
        # If no location provided, place the agent at the opposite side for the next trial
        self.agent_pos = random.sample([self.goal_obj_up.cur_pos, self.goal_obj_down.cur_pos], 1)[0]
        if starting_location is not None:
            self.agent_pos = starting_location
            reverse_direction = self.agent_pos[0] == self.goal_obj_up.cur_pos[0]
        if self.agent_pos[1] <= self.grid.height // 2:
            self.agent_dir = 1
        else:
            self.agent_dir = 3
        if self.cue_obj.cur_pos is not None:
            self.grid.set(*self.cue_obj.cur_pos, Wall())
            self.grid.set(self.cue_obj.cur_pos[0], self.cue_obj.cur_pos[1] + 2, Wall())
            self.cue_obj.cur_pos = None

        if self.trial_count < self.nb_trial_per_episode:
            self.start_next_trial(reverse_direction)
        if info is not None:
            return info

    def goals_appear(self):
        if self.hallway_start > self.hallway_end: # going from right to left
            x_pos_goal = self.hallway_end - 1
        else:
            x_pos_goal = self.hallway_end + 1
        pos0 = (x_pos_goal, self.height // 2 - 1)
        pos1 = (x_pos_goal, self.height // 2 + 1)
        self.put_obj(self.goal_obj_up, *pos0)
        self.put_obj(self.goal_obj_down, *pos1)

    def modify_context_cue_next_to_goals(self, goal1_pos, goal2_pos, modification):
        y_pos_sidestep = 1
        if modification == 'place':
            self.grid.set(goal1_pos[0], goal1_pos[1]-y_pos_sidestep, self.cue_obj)
            self.grid.set(goal2_pos[0], goal2_pos[1]+y_pos_sidestep, self.cue_obj)
        elif modification == 'remove':
            self.grid.set(goal1_pos[0], goal1_pos[1]+y_pos_sidestep, Wall())
            self.grid.set(goal2_pos[0], goal2_pos[1]-y_pos_sidestep, Wall())
            if self.cue_obj.cur_pos is not None:
                self.grid.set(*self.cue_obj.cur_pos, Wall())
                self.grid.set(self.cue_obj.cur_pos[0], self.cue_obj.cur_pos[1] + 2, Wall())
                self.cue_obj.cur_pos = None


    def create_goals(self):
        if self.changing_reward == 'monkey':
            goal_colors = self.monkey_data[self.index_preset]['goal_colors'][self.trial_count]
        else:
            goal_colors = self._rand_subset(self.goal_colors, 2)
        other_objs = [Ball(goal_colors[0]), Ball(goal_colors[1])]
        self.goal_obj_up = other_objs[0]
        self.goal_obj_down = other_objs[1]

    def reset(self):
        self.trial_step_count = 0
        obs = MiniGridEnv.reset(self)
        obs['reward'] = 0.0
        obs['action'] = 2 # give by default: forward
        if isinstance(self.changing_reward, int):
            obs['task_index'] = self.index_preset
        elif self.changing_reward == 'monkey':
            obs['task_index'] = self.find_task_index(self.goal_colors)
        return obs

    def add_side_indicator(self):
        # Add visual indicators that differentiate the four arms of the maze
        upper_room_wall = self.height // 2 - 2
        lower_room_wall = self.height // 2 + 2
        self.put_obj(Box('red'), self.hallway_start, upper_room_wall + 1)
        self.put_obj(Box('yellow'), self.hallway_start, lower_room_wall - 1)
        self.put_obj(Box('purple'), self.hallway_end, upper_room_wall + 1)
        self.put_obj(Box('blue'), self.hallway_end, lower_room_wall - 1)

    def add_distal_cue(self):
        upper_room_wall = self.height // 2 - 2
        lower_room_wall = self.height // 2 + 2
        self.put_obj(Box('red'), self.height // 2, upper_room_wall)
        self.put_obj(Box('blue'), self.height // 2, lower_room_wall)

class Square(WorldObj):
    def __init__(self, color='blue'):
        temp_color = color
        if not isinstance(color, str):
            temp_color = 'blue'
        super(Square, self).__init__('square', color=temp_color)
        self.color = color

    def see_behind(self):
        return False

    def encode(self):
        """
        Override to encode object that have RGB color, normally need to be an int from the COLOR_TO_IDX dict
        Done for Visualization. Unclear how it would affect the agent.
        """
        """Encode the a description of this object as a 3-tuple of integers"""

        def getRGBfromInteger(RGBint):
            blue = RGBint & 255
            green = (RGBint >> 8) & 255
            red = (RGBint >> 16) & 255
            return red, green, blue

        def getIntegerfromRGB(rgb):
            red = rgb[0]
            green = rgb[1]
            blue = rgb[2]
            RGBint = (red<<16) + (green<<8) + blue
            return RGBint

        if not isinstance(self.color, str):
            int_color = getIntegerfromRGB(self.color)
            normalized_color = int_color / 16777215 # (x-min)/(max-min) where min int=0 and max int=16777215
            return (OBJECT_TO_IDX[self.type], normalized_color, 0)
        else:
            return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    def render(self, img):
        if not isinstance(self.color, str): # already RGB np.array
            fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)
        else:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, color):
        super(Box, self).__init__('box', color)

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), (100,100,100))
        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (100,100,100))
        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

class AssociativeMemoryS17Random(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=17, random_length=True)

register(
    id='MiniGrid-Associative-MemoryS17Random-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS17Random',
)

class AssociativeMemoryS13Random(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=13, random_length=True)

register(
    id='MiniGrid-Associative-MemoryS13Random-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS13Random',
)

class AssociativeMemoryS13(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=13)

register(
    id='MiniGrid-Associative-MemoryS13-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS13',
)

class AssociativeMemoryS11(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=11)

register(
    id='MiniGrid-Associative-MemoryS11-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS11',
)

class AssociativeMemoryS9(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=9)

register(
    id='MiniGrid-Associative-MemoryS9-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS9',
)

class AssociativeMemoryS9R2(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=9, changing_reward=2, nb_trial_per_episode=2500000)

register(
    id='MiniGrid-Associative-MemoryS9R2-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS9R2',
)

class AssociativeMemoryS9R2SI(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=9, changing_reward=2, nb_trial_per_episode=2500000, side_indicator=True)

register(
    id='MiniGrid-Associative-MemoryS9R2SI-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS9R2SI',
)

class AssociativeMemoryS9RSI(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=9, changing_reward=True, nb_trial_per_episode=2500, side_indicator=True) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS9RSI-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS9RSI',
)

class AssociativeMemoryS9RSI100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=9, changing_reward=True, nb_trial_per_episode=100, side_indicator=True, max_steps=50000)

register(
    id='MiniGrid-Associative-MemoryS9RSI100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS9RSI100',
)

class AssociativeMemoryS9RSI1000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=9, changing_reward=True, nb_trial_per_episode=1000, side_indicator=True, max_steps=80000)

register(
    id='MiniGrid-Associative-MemoryS9RSI1000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS9RSI1000',
)

class AssociativeMemoryS7RSI10k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=10000, side_indicator=True, max_steps=200000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RSI10k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RSI10k',
)

class AssociativeMemoryS7RSI5000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=5000, side_indicator=True, max_steps=100000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RSI5000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RSI5000',
)

class AssociativeMemoryS7RSI1000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=1000, side_indicator=True, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RSI1000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RSI1000',
)

class AssociativeMemoryS7RSI500(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=500, side_indicator=True, max_steps=40000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RSI500-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RSI500',
)

class AssociativeMemoryS7RSI200(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=200, side_indicator=True, max_steps=20000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RSI200-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RSI200',
)

class AssociativeMemoryS7RSI100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=100, side_indicator=True, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RSI100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RSI100',
)

class AssociativeMemoryS7R100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=100, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R100',
)


class AssociativeMemoryS7R2SI1000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=2, nb_trial_per_episode=1000, side_indicator=True, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R2SI1000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R2SI1000',
)

## Changed reward hierarchy


##  50 trials per episode
class AssociativeMemoryS7R2T50(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=2, nb_trial_per_episode=50, side_indicator=False, max_steps=4000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R2T50-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R2T50',
)

class AssociativeMemoryS7R4T50(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=4, nb_trial_per_episode=50, side_indicator=False, max_steps=4000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R4T50-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R4T50',
)

class AssociativeMemoryS7R8T50(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=8, nb_trial_per_episode=50, side_indicator=False, max_steps=4000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R8T50-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R8T50',
)

class AssociativeMemoryS7R18T50(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=18, nb_trial_per_episode=50, side_indicator=False, max_steps=4000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R18T50-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R18T50',
)

class AssociativeMemoryS7R31T50(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=31, nb_trial_per_episode=50, side_indicator=False, max_steps=4000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R31T50-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R31T50',
)

class AssociativeMemoryS7RT50(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=50, side_indicator=False, max_steps=4000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RT50-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RT50',
)



##  100 trials per episode
class AssociativeMemoryS7R2T100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=2, nb_trial_per_episode=100, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R2T100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R2T100',
)

class AssociativeMemoryS7R4T100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=4, nb_trial_per_episode=100, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R4T100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R4T100',
)

class AssociativeMemoryS7R8T100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=8, nb_trial_per_episode=100, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R8T100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R8T100',
)
class AssociativeMemoryS7R18T100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=18, nb_trial_per_episode=100, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R18T100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R18T100',
)

class AssociativeMemoryS7R31T100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=31, nb_trial_per_episode=100, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R31T100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R31T100',
)

class AssociativeMemoryS7RT100(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=100, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RT100-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RT100',
)

##  500 trials per episode
class AssociativeMemoryS7R2T500(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=2, nb_trial_per_episode=500, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R2T500-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R2T500',
)

class AssociativeMemoryS7R4T500(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=4, nb_trial_per_episode=500, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R4T500-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R4T500',
)

class AssociativeMemoryS7R8T500(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=8, nb_trial_per_episode=500, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R8T500-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R8T500',
)
class AssociativeMemoryS7R18T500(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=18, nb_trial_per_episode=500, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R18T500-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R18T500',
)

class AssociativeMemoryS7RT500(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=500, side_indicator=False, max_steps=8000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RT500-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RT500',
)


## 1k trials per episode
class AssociativeMemoryS7R2T1000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=2, nb_trial_per_episode=1000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R2T1000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R2T1000',
)

class AssociativeMemoryS7R4T1000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=4, nb_trial_per_episode=1000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R4T1000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R4T1000',
)

class AssociativeMemoryS7R8T1000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=8, nb_trial_per_episode=1000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R8T1000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R8T1000',
)
class AssociativeMemoryS7R18T1000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=18, nb_trial_per_episode=1000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R18T1000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R18T1000',
)

class AssociativeMemoryS7RT1000(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=1000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RT1000-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RT1000',
)


## 5k trials per episode
class AssociativeMemoryS7R2T5k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=2, nb_trial_per_episode=5000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R2T5k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R2T5k',
)

class AssociativeMemoryS7R4T5k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=4, nb_trial_per_episode=5000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R4T5k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R4T5k',
)

class AssociativeMemoryS7R8T5k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=8, nb_trial_per_episode=5000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R8T5k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R8T5k',
)
class AssociativeMemoryS7R18T5k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=18, nb_trial_per_episode=5000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R18T5k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R18T5k',
)

class AssociativeMemoryS7RT5k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=5000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RT5k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RT5k',
)



## 10k trials per episode
class AssociativeMemoryS7R2T10k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=2, nb_trial_per_episode=10000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R2T10k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R2T10k',
)

class AssociativeMemoryS7R4T10k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=4, nb_trial_per_episode=10000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R4T10k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R4T10k',
)

class AssociativeMemoryS7R8T10k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=8, nb_trial_per_episode=10000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R8T10k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R8T10k',
)
class AssociativeMemoryS7R18T10k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=18, nb_trial_per_episode=10000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R18T10k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R18T10k',
)

class AssociativeMemoryS7RT10k(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=True, nb_trial_per_episode=10000, side_indicator=False, max_steps=80000) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7RT10k-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RT10k',
)


## 1 Trial per episode
class AssociativeMemoryS7R2T1(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=2, nb_trial_per_episode=1, side_indicator=False, max_steps=80) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R2T1-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R2T1',
)

class AssociativeMemoryS7R4T1(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=4, nb_trial_per_episode=1, side_indicator=False, max_steps=80) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R4T1-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R4T1',
)

class AssociativeMemoryS7R8T1(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=8, nb_trial_per_episode=1, side_indicator=False, max_steps=80) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R8T1-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R8T1',
)
class AssociativeMemoryS7R18T1(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=18, nb_trial_per_episode=1, side_indicator=False, max_steps=80) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R18T1-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R18T1',
)

class AssociativeMemoryS7R31T1(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=31, nb_trial_per_episode=1, side_indicator=False, max_steps=80) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R31T1-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R31T1',
)

class AssociativeMemoryS7R31T10(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=31, nb_trial_per_episode=10, side_indicator=False, max_steps=800) # nb_trial_per_episode=200000

register(
    id='MiniGrid-Associative-MemoryS7R31T10-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R31T10',
)

class AssociativeMemoryS7R31T10CL(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=31, nb_trial_per_episode=10, side_indicator=False, max_steps=80000, curriculum_learning=True)

register(
    id='MiniGrid-Associative-MemoryS7R31T10CL-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R31T10CL',
)

class AssociativeMemoryS7R31T200(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward=31, nb_trial_per_episode=200, side_indicator=False, max_steps=80000, curriculum_learning=False)

register(
    id='MiniGrid-Associative-MemoryS7R31T200-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7R31T200',
)


class AssociativeMemoryS7RMTM(AssociativeMemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=7, changing_reward='monkey', nb_trial_per_episode='monkey', side_indicator=False, max_steps=800000)

register(
    id='MiniGrid-Associative-MemoryS7RMTM-v0',
    entry_point='gym_minigrid.envs:AssociativeMemoryS7RMTM',
)