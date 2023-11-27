import torch.nn as nn
import torch.nn.functional as F
import torch as th

def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

class MLPAgent1(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent1, self).__init__()
        self.args = args
        self.input_shape = input_shape
        # self.state_shape = self.input_shape-self.args.zs_dim

        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.ln1 = nn.LayerNorm(args.rnn_hidden_dim)

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.ln2 = nn.LayerNorm(args.rnn_hidden_dim)

        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions=None):
        x = self.fc1(inputs.view(-1, self.input_shape))
        x = AvgL1Norm(x)
        # x = th.cat([x,inputs[:,self.state_shape:].view(-1, self.args.zs_dim)],dim=-1)
        # x = self.ln1(x)
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))
        # x = self.ln2(x)

        if self.agent_return_logits:
            actions = self.fc4(x)
        else:
            actions = F.tanh(self.fc4(x))
        return {"actions": actions, "hidden_state": hidden_state}


