import torch as th
import torch.nn as nn
import torch.nn.functional as F
# device = th.device("cpu")
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class FACMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACCritic, self).__init__()
        self.input_obs = scheme["obs"]["vshape"]
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self.input_obs + self.n_actions
        self.output_type = "q"
        self.hidden_states = None
        self.input_shape2 = args.zs_dim*2 + args.rnn_hidden_dim

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim).to(device)
        self.ln1 = nn.LayerNorm(args.rnn_hidden_dim).to(device)
        self.fc2 = nn.Linear(self.input_shape2, args.rnn_hidden_dim).to(device)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim).to(device)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, 1).to(device)

        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, zsa, hidden_state=None):
        if actions is not None:
            # input_obs = self.input_obs
            inputs_obs_action = th.cat([inputs[:,:self.input_obs].view(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
            inputs_obs_action = inputs_obs_action.to(device)
            embe = th.cat([inputs[:,self.input_obs:].view(-1, self.args.zs_dim),
                             zsa.reshape(-1, self.args.zs_dim)], dim=-1).to(device)
        x = self.ln1(self.fc1(inputs_obs_action))
        x = th.cat([x,embe],dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.fc4(x)

        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"] + scheme["zs"]["vshape"][0]

        return input_shape


class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape