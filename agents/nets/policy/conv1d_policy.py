import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

class Conv1dPolicy(nn.Module):
    def __init__(self, state_shape, num_of_actions, body_fets=((1, 4), (4, 8), (8, 16), (16, 16))):
        super(Conv1dPolicy, self).__init__()
        _, w = state_shape
        body = [self._conv_block(in_ch, out_ch)
                for (in_ch, out_ch) in body_fets]
        self.body = nn.Sequential(*body)
        head = nn.Sequential(nn.Linear(w * 16, num_of_actions), nn.ReLU())
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution