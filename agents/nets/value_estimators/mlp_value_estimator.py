import torch.nn as nn

class MLPValueEstimator(nn.Module):
    def __init__(self,
                 state_shape,
                 body_fets = (32, 64, 32, 16, 8),
                 num_actions = 4):
        self.in_state_shape = state_shape
        super(MLPValueEstimator, self).__init__()
        body = [nn.Sequential(nn.Linear(in_fets, out_fets), nn.ReLU())
                for in_fets, out_fets in zip(body_fets, body_fets[1:])]
        self.body = nn.Sequential(*body)
        self.head = nn.Sequential(nn.Linear(body_fets[-1], num_actions), nn.ReLU())

    def forward(self, x):
        x = self.body(x)
        x = self.head(x.view(x.size(0), -1))
        return x
