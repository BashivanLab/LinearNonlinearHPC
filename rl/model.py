import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

def init_params(m):
    # classname = m.__class__.__name__
    # if classname.find("Linear") != -1:
    if isinstance(m, torch.nn.modules.linear.Linear):
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class LinearLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearLSTMCell, self).__init__()

        # Follow LSTMCell definition from https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
        # But remove any non-linearity

        self.hidden_size = hidden_size
        # self.bn = nn.BatchNorm1d(hidden_size)

        # Weights for Input Gate
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)

        # Weights for Forget Gate
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)

        # Weights for Cell Update
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)

        # Weights for Output Gate
        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)

        # self.layernorm_i = nn.LayerNorm(hidden_size)
        # self.layernorm_f = nn.LayerNorm(hidden_size)
        # self.layernorm_g = nn.LayerNorm(hidden_size)
        # self.layernorm_o = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, hidden):
        # separate tuple containing hidden and cell state
        h, c = hidden

        # # Input Gate (Linear version, without sigmoid)
        # i = self.W_ii(x) + self.W_hi(h)
        # # Forget Gate (Linear version, without sigmoid)
        # f = self.W_if(x) + self.W_hf(h)
        # # Cell Update (Linear version, without tanh)
        # g = self.W_ig(x) + self.W_hg(h)
        # # Output Gate (Linear version, without sigmoid)
        # o = self.W_io(x) + self.W_ho(h)

        # i = self.bn(self.W_ii(x) + self.W_hi(h)) # Input Gate without sigmoid
        # f = self.bn(self.W_if(x) + self.W_hf(h)) # Forget Gate without sigmoid
        # g = self.bn(self.W_ig(x) + self.W_hg(h)) # Cell Update without tanh
        # o = self.bn(self.W_io(x) + self.W_ho(h)) # Output Gate without sigmoid

        # # Compute gates with linear transformations
        # i = self.layernorm_i(self.W_ii(x) + self.W_hi(h))  # Input Gate without sigmoid
        # f = self.layernorm_f(self.W_if(x) + self.W_hf(h))  # Forget Gate without sigmoid
        # g = self.layernorm_g(self.W_ig(x) + self.W_hg(h))  # Cell Update without tanh
        # o = self.layernorm_o(self.W_io(x) + self.W_ho(h))  # Output Gate without sigmoid

        i = self.sigmoid(self.W_ii(x) + self.W_hi(h))  # Input Gate without sigmoid
        f = self.sigmoid(self.W_if(x) + self.W_hf(h))  # Forget Gate without sigmoid
        g = self.relu(self.W_ig(x) + self.W_hg(h))  # Cell Update without tanh
        o = self.sigmoid(self.W_io(x) + self.W_ho(h))  # Output Gate without sigmoid

        # Cell State Update
        new_cell_state = f * c + i * g

        # Hidden State Update (Linear combination, no tanh)
        new_hidden_state = o * self.relu(new_cell_state)
        return (new_hidden_state, new_cell_state)


class LinearRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear transformation for input and hidden state
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)  # W_ih^T and b_ih
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)  # W_hh^T and b_hh
        self.layernorm = nn.LayerNorm(hidden_size)
        # self.relu = nn.ReLU()

    def forward(self, x, hidden):
        h_prev, c = hidden # always receive and return a 'c' to be compatible with code for LSTM
        # h_t = self.W_ih(x) + self.W_hh(h_prev)
        h_t = self.layernorm(self.W_ih(x) + self.W_hh(h_prev))
        # h_t = torch.tanh(self.W_ih(x) + self.W_hh(h_prev)) # Nonlinear version for debugging
        # h_t = self.relu(self.W_ih(x) + self.W_hh(h_prev))

        return h_t, c

class UGRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UGRNN, self).__init__()
        # https://openreview.net/pdf?id=BydARw9ex
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ch = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_cx = nn.Linear(input_size, hidden_size, bias=True)
        self.W_gh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_gx = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layernorm_c = nn.LayerNorm(hidden_size)
        self.layernorm_g = nn.LayerNorm(hidden_size)
        # self.layernorm_h = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden # always receive and return a 'c' to be compatible with code for LSTM

        ## Withour layer norm
        # c = self.relu(self.W_ch(h_prev) + self.W_cx(x))
        # g_t = self.sigmoid(self.W_gh(h_prev) + self.W_gx(x))
        # h_t = g_t*h_prev + (1-g_t)*c

        c = self.relu(self.layernorm_c(self.W_ch(h_prev) + self.W_cx(x)))
        g_t = self.sigmoid(self.layernorm_g(self.W_gh(h_prev) + self.W_gx(x)))
        h_t = g_t * h_prev + (1 - g_t) * c
        # h_t = self.layernorm_h(g_t * h_prev + (1 - g_t) * c)
        return h_t, c


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_rnn_memory=False, use_text=False, use_transition_as_input=False, nb_tasks=False, linear_rnn=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_rnn_memory = use_rnn_memory
        self.linear_rnn = linear_rnn
        self.use_transition_as_input = use_transition_as_input
        if nb_tasks == 'monkey':
            self.nb_tasks = 31
        else:
            self.nb_tasks = nb_tasks
        self.name = 'default'

        # Define image embedding
        if obs_space["image"][0] <= 5: # Smaller convnet if observation/image/input is 5x5 or smaller
            kernel_size = (2, 2)
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size),
                nn.ReLU(),
                # nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, kernel_size),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size),
                nn.ReLU()
            )
            self.image_embedding_size = 256
        else:
            kernel_size = (2, 2)
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, kernel_size),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size),
                nn.ReLU()
            )
            n = obs_space["image"][0]
            m = obs_space["image"][1]
            self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64


        # Define memory
        if self.use_rnn_memory:
            if use_transition_as_input:
                # Increase input dimension for reward
                # self.memory_rnn = nn.LSTMCell(self.image_embedding_size + 1, self.semi_rnn_memory_size)

                # Increase input dimension to add the reward and previous action (one hot)
                self.nb_actions = action_space.n # needed to one-hot the actions
                if self.linear_rnn:
                    self.memory_rnn = LinearRNNCell(self.image_embedding_size + 1 + self.nb_actions + self.nb_tasks,
                                                    self.semi_rnn_memory_size)
                    # self.memory_rnn = LinearRNNCell(self.image_embedding_size + 1 + self.nb_actions + self.nb_tasks,
                    #                                 self.semi_rnn_memory_size)
                    # self.memory_rnn = LinearLSTMCell(self.image_embedding_size + 1 + self.nb_actions + self.nb_tasks,
                    #                                 self.semi_rnn_memory_size)
                else:
                    self.memory_rnn = nn.LSTMCell(self.image_embedding_size + 1 + self.nb_actions + self.nb_tasks,
                                                  self.semi_rnn_memory_size)
            else:
                if self.linear_rnn:
                    self.memory_rnn = LinearRNNCell(self.image_embedding_size, self.semi_rnn_memory_size)
                    # self.memory_rnn = LinearRNNCell(self.image_embedding_size, self.semi_rnn_memory_size)
                    # self.memory_rnn = LinearLSTMCell(self.image_embedding_size, self.semi_rnn_memory_size)
                else:
                    self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_rnn_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_rnn_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)
        # if self.linear_rnn:
        #     self.apply(init_weights)
        # else:
        #     self.apply(init_params)

    @property
    def rnn_memory_size(self):
        return 2*self.semi_rnn_memory_size

    @property
    def semi_rnn_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, rnn_memory):
        # rnn_memory_copy_for_debug = rnn_memory.clone()
        x = obs.image.transpose(1, 3).transpose(2, 3) # x becomes of shape: batch size, obj_id/color_id/states, x, y
        # print('cnn input', x.shape)
        x = self.image_conv(x)
        # print('cnn output', x.shape)
        x = x.reshape(x.shape[0], -1) # transform cnn output (batch_size, nb kernel) into (batch_size, input size) for the following linear layers
        # print('cnn output after reshape', x.shape)
        # print('image embedding size', self.image_embedding_size) # looks like image_embedding_size == x.shape[1]

        if self.use_rnn_memory:
            hidden = (rnn_memory[:, :self.semi_rnn_memory_size], rnn_memory[:, self.semi_rnn_memory_size:]) # (hidden state, cell state)
            if self.use_transition_as_input:
                # concatenate reward and action to x. Change shape from e.g. (16 (batch_size), 256(lstm_input_size)) to (16,257)
                # x = torch.cat((x, obs.reward.unsqueeze(1)), dim=1)

                x = torch.cat((x, obs.reward.unsqueeze(1), F.one_hot(obs.action.long(), self.nb_actions),  F.one_hot(obs.task_index.long(), self.nb_tasks)), dim=1)
                # print(f'Settings:{torch.unique(obs.task_index, return_counts=True)[0]}; Distribution: {torch.unique(obs.task_index, return_counts=True)[1]}')

                # if torch.sum(obs.reward > 0) > int(obs.reward.shape[0] * 0.3): #(obs.reward > 0).any():
                #     print()
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            rnn_memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        # print(torch.isnan(rnn_memory).any())
        # print(torch.isnan(x).any())
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, rnn_memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
