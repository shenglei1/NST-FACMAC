import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)

class Encoder(nn.Module):
    def __init__(self, scheme, args): #state_dim, action_dim):
        super(Encoder, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape1 = scheme["obs"]["vshape"]
        self.input_shape2 = args.zs_dim + self.n_actions

        # state encoder
        self.fc1 = nn.Linear(self.input_shape1, args.rnn_hidden_dim).to(device)
        # self.ln1 = nn.LayerNorm(args.rnn_hidden_dim).to(device)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim).to(device)
        # self.ln2 = nn.LayerNorm(args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.zs_dim).to(device)
        # self.ln3 = nn.LayerNorm(args.zs_dim).to(device)


        # state-action encoder
        self.fc11 = nn.Linear(self.input_shape2, args.rnn_hidden_dim).to(device)
        self.fc21 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim).to(device)
        self.fc31 = nn.Linear(args.rnn_hidden_dim, args.zs_dim).to(device)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def zs(self, state):#, hidden_state):
        zs = F.relu(self.fc1(state))
        # zs = F.relu(zs)
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # h = self.rnn(zs, h_in)

        zs = F.relu(self.fc2(zs))
        zs = AvgL1Norm(self.fc3(zs))
        # zs = self.ln3(self.fc3(zs))
        return zs


    def zsa(self, zs_action):
        zsa = F.relu(self.fc11(zs_action))
        zsa = F.relu(self.fc21(zsa))
        zsa = self.fc31(zsa)
        return zsa
