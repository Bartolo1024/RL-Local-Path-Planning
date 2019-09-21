import math
import torch.nn as nn
from agents.nets import utils


class Conv2dValueEstimator(nn.Module):
    def __init__(self, state_shape, head_fets=((512,)), num_actions=4):
        super(Conv2dValueEstimator, self).__init__()
        in_channels, h, w = state_shape
        body = [utils._conv2d_block(in_ch, out_ch)
                for (in_ch, out_ch) in ((in_channels, 32), (32, 48), (48, 48), (48, 64), (64, 64))]
        h = math.ceil(h / 2 ** len(body))
        w = math.ceil(w / 2 ** len(body))
        self.body = nn.Sequential(*body)
        head = [nn.Sequential(nn.Linear(in_fet, out_fet), nn.ReLU())
                for in_fet, out_fet in [(64 * h * w) + head_fets, (head_fets[-1], num_actions)]]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x.view(x.size(0), -1))
        return x