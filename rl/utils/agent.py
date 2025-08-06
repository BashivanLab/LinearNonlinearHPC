import torch

import utils
from .other import device
from model import ACModel
from epn import EPNModel
from lstm_memorygame import LstmMemGameModel
from epn_assocmem import EPNAssocModel
from model_relu_lstm import ACModel as ReluLSTMACModel
from model_fully_relu_lstm import ACModel as FullyReluLSTMACModel
from model_relu_rnn import ACModel as ReluRNNACModel
from model_relu_ugrnn_layernorm import ACModel as UgrnnLayernormACModel
from model_relu_ugrnn import ACModel as UgrnnACModel
from model_relu_gru import ACModel as GRUACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_rnn_memory=False, use_text=False, use_transition_as_input=False,
                 model_type=None, nb_tasks=False, checkpoint_id=None, linear_rnn=False):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        if model_type == 'epn':
            self.acmodel = EPNModel(obs_space, action_space)
        elif model_type == 'lstm_memorygame':
            self.acmodel = LstmMemGameModel(obs_space, action_space, use_rnn_memory=use_rnn_memory)
        elif model_type == 'epn_assocmem':
            self.acmodel = EPNAssocModel(obs_space, action_space, nb_tasks=nb_tasks)

        elif model_type == 'ugrnn':
            self.acmodel = UgrnnACModel(obs_space, action_space, use_rnn_memory=use_rnn_memory, use_text=use_text,
                                   use_transition_as_input=use_transition_as_input, nb_tasks=nb_tasks, linear_rnn=linear_rnn)
        elif model_type == 'ugrnn_layernorm':
            self.acmodel = UgrnnLayernormACModel(obs_space, action_space, use_rnn_memory=use_rnn_memory, use_text=use_text,
                                   use_transition_as_input=use_transition_as_input, nb_tasks=nb_tasks, linear_rnn=linear_rnn)
        elif model_type == 'relu_lstm':
            self.acmodel = ReluLSTMACModel(obs_space, action_space, use_rnn_memory=use_rnn_memory, use_text=use_text,
                                   use_transition_as_input=use_transition_as_input, nb_tasks=nb_tasks, linear_rnn=linear_rnn)
        elif model_type == 'relu_rnn':
            self.acmodel = ReluRNNACModel(obs_space, action_space, use_rnn_memory=use_rnn_memory, use_text=use_text,
                                   use_transition_as_input=use_transition_as_input, nb_tasks=nb_tasks, linear_rnn=linear_rnn)
        elif model_type == 'relu_gru':
            self.acmodel = GRUACModel(obs_space, action_space, use_rnn_memory=use_rnn_memory, use_text=use_text,
                                   use_transition_as_input=use_transition_as_input, nb_tasks=nb_tasks, linear_rnn=linear_rnn)
        elif model_type == 'fully_relu_lstm':
            self.acmodel = FullyReluLSTMACModel(obs_space, action_space, use_rnn_memory=use_rnn_memory, use_text=use_text,
                                   use_transition_as_input=use_transition_as_input, nb_tasks=nb_tasks, linear_rnn=linear_rnn)
        else:
            self.acmodel = ACModel(obs_space, action_space, use_rnn_memory=use_rnn_memory, use_text=use_text,
                                   use_transition_as_input=use_transition_as_input, nb_tasks=nb_tasks, linear_rnn=linear_rnn)

        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.rnn_memories = torch.zeros(self.num_envs, self.acmodel.rnn_memory_size, device=device)
        if 'epn' in self.acmodel.name:
            self.memories = torch.zeros(self.num_envs, self.acmodel.capacity, self.acmodel.memory_size, device=device)
            self.memory_indexes = torch.zeros(self.num_envs, device=device, dtype=torch.long)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir, checkpoint_id))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir, checkpoint_id))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if 'epn' in self.acmodel.name:
                dist, _, self.memories, self.memory_indexes = self.acmodel(preprocessed_obss, self.memories, self.memory_indexes)
            elif self.acmodel.recurrent:
                dist, _, self.rnn_memories = self.acmodel(preprocessed_obss, self.rnn_memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            # actions = dist.probs.max(1, keepdim=True)[1]
            actions = dist.probs.max(1, keepdim=False)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
        if self.acmodel.recurrent:
            self.rnn_memories *= masks
        if 'epn' in self.acmodel.name:
            self.memories *= masks
            self.memory_indexes *= masks.squeeze().long()

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
