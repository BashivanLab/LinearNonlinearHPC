from abc import ABC, abstractmethod
import torch
import numpy as np
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if 'epn' in self.acmodel.name:
            self.memory = torch.zeros(shape[1], self.acmodel.capacity, self.acmodel.memory_size, device=self.device)
            self.memory_index = torch.zeros(shape[1], device=self.device, dtype=torch.long) # next available index to write a memory in
            self.memories = torch.zeros(*shape, self.acmodel.capacity, self.acmodel.memory_size, device=self.device)
            self.memory_indexes = torch.zeros(*shape, device=self.device, dtype=torch.long)
        elif self.acmodel.recurrent:
        # if self.acmodel.recurrent:
            self.rnn_memory = torch.zeros(shape[1], self.acmodel.rnn_memory_size, device=self.device)
            self.rnn_memories = torch.zeros(*shape, self.acmodel.rnn_memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        optimal = []
        incomplete_trials = []
        average_nb_trial_per_episode = []

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            # self.obs: list of length batch_size. Each element: dict['image', 'direction', 'mission']
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                # if self.acmodel.name == 'epn':
                if 'epn' in self.acmodel.name:
                    dist, value, memory, memory_index = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1).repeat(1, self.acmodel.capacity).unsqueeze(2), self.memory_index * self.mask)
                # elif self.acmodel.name == 'epn_assocmem':
                #     dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1).repeat(1, self.acmodel.capacity).unsqueeze(2))
                elif self.acmodel.recurrent:
                    dist, value, rnn_memory = self.acmodel(preprocessed_obs, self.rnn_memory * self.mask.unsqueeze(1)) # * self.mask.unsqueeze(1): reset the memory of env that were done
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()

            obs, reward, done, info = self.env.step(action.cpu().numpy())

            # Log ratio of optimal choice
            final_rewards = (np.asarray(reward) > 0)
            if final_rewards.any():
                final_rewards_index = final_rewards.nonzero()[0]
                if 'is_optimal' in np.asarray(info)[final_rewards_index[0]].keys(): # skip for Tasks that don't have this info
                    for choice in np.asarray(info)[final_rewards_index]:
                        optimal.append(choice['is_optimal'])

            # Log whether the trials are completed (i.e. reach a goal) or was reset because the agent reached the
            # maximum of step for a trial (e.g. agent spinning instead of doing the task)
            has_info_on_incomplete_trial = ['incomplete_trial' in e.keys() for e in info]
            if any(has_info_on_incomplete_trial):
                for info_on_incomplete_trial in np.asarray(info)[has_info_on_incomplete_trial]:
                    incomplete_trials.append(info_on_incomplete_trial['incomplete_trial'])

            # Log the average number of trial per episodes across the different parallel envs
            average_nb_trial_per_episode.append(np.mean([e['nb_trial_per_episode'] for e in info]))

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if 'epn' in self.acmodel.name:
                self.memories[i] = self.memory
                self.memory_indexes[i] = self.memory_index
                self.memory = memory
                self.memory_index = memory_index
            if self.acmodel.recurrent:
                self.rnn_memories[i] = self.rnn_memory
                self.rnn_memory = rnn_memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if 'epn' in self.acmodel.name:
                _, next_value, _, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1).repeat(1, self.acmodel.capacity).unsqueeze(2), self.memory_index * self.mask)
            elif self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.rnn_memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # if self.acmodel.name == 'epn':
        if 'epn' in self.acmodel.name:
            # T x P x C x D -> P x T x C x D -> (P * T) x C x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
            exps.memory_index = self.memory_indexes.transpose(0, 1).reshape(-1).unsqueeze(1)
        elif self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.rnn_memory = self.rnn_memories.transpose(0, 1).reshape(-1, *self.rnn_memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "optimal_choice": optimal,
            "incomplete_trials": incomplete_trials,
            'average_nb_trial_per_episode': average_nb_trial_per_episode,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
