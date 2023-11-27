import torch as th
import torch.nn as nn
import torch.nn.functional as F


class td3Critic(nn.Module):
    def __init__(self, scheme, args):
        super(td3Critic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents

        self.output_type = "q"

        # Set up network layers
        # Q1 architecture
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc5 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc6 = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions * self.n_agents),
                             actions.contiguous().view(-1, self.n_actions * self.n_agents)], dim=-1)
        x11 = F.relu(self.fc1(inputs))
        x12 = F.relu(self.fc2(x11))
        q1 = self.fc3(x12)

        x21 = F.relu(self.fc4(inputs))
        x22 = F.relu(self.fc5(x21))
        q2 = self.fc6(x22)
        return q1,q2, hidden_state

    def Q1(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions * self.n_agents),
                             actions.contiguous().view(-1, self.n_actions * self.n_agents)], dim=-1)
        x11 = F.relu(self.fc1(inputs))
        x12 = F.relu(self.fc2(x11))
        q1 = self.fc3(x12)
        return q1, hidden_state

    def _get_input_shape(self, scheme):
        # The centralized critic takes the state input, not observation
        input_shape = scheme["state"]["vshape"]
        return input_shape