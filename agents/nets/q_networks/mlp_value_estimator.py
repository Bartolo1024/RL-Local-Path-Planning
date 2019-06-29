import torch
import torch.nn as nn

class MLPValueEstimator(nn.Module):
    def __init__(self,
                 state_shape,
                 body_fets = (64, 32, 16, 8),
                 num_actions = 3):
        super(MLPValueEstimator, self).__init__()
        self.stump = nn.Sequential(nn.Linear(state_shape[1] + 2, body_fets[0]), nn.ReLU())
        body = [nn.Sequential(nn.Linear(in_fets, out_fets), nn.ReLU())
                for in_fets, out_fets in zip(body_fets, body_fets[1:])]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(body_fets[-1], num_actions)

    def forward(self, ranges, target_points):
        x = [torch.cat((sens, tp.unsqueeze(0)), 1) for sens, tp in zip(ranges, target_points)]
        x = torch.stack(tuple(x))
        x = self.stump(x)
        x = self.body(x)
        x = self.head(x.view(x.size(0), -1))
        return x
