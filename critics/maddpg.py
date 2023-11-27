import torch as th
import torch.nn as nn
import torch.nn.functional as F

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_state = scheme["state"]["vshape"]
        self.input_shape = self.input_state + self.n_actions * self.n_agents
        self.input_shape2 = args.zs_dim * 2 * self.n_agents + args.rnn_hidden_dim
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)
        self.ln1 = nn.LayerNorm(args.rnn_hidden_dim).to(device)
        self.fc2 = nn.Linear(self.input_shape2, args.rnn_hidden_dim).to(device)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim).to(device)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, 1).to(device)

    def forward(self, inputs, actions, zsa, hidden_state=None):
        if actions is not None:
            inputs_state_acts = th.cat([inputs[:,:self.input_state].reshape(-1, self.input_shape - self.n_actions * self.n_agents),
                             actions.contiguous().view(-1, self.n_actions * self.n_agents)], dim=-1)
            embe = th.cat([inputs[:, self.input_state:].view(-1, self.args.zs_dim* self.n_agents),zsa.reshape(-1, self.args.zs_dim* self.n_agents)], dim=-1).to(device)
        # x = self.ln1(self.fc1(inputs_state_acts))
        x = AvgL1Norm(self.fc1(inputs_state_acts))
        x = th.cat([x, embe], dim=1)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        # The centralized critic takes the state input, not observation
        input_shape = scheme["state"]["vshape"]
        return input_shape