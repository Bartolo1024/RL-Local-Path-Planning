import torch
import torch.nn as nn
import sys

class MLPValueEstimator(nn.Module):
    def __init__(self,
                 state_shape,
                 body_fets=(64, 32, 16, 8),
                 num_actions=3,
                 target_point_dim=2):
        super(MLPValueEstimator, self).__init__()
        self.stump = nn.Sequential(nn.Linear(state_shape[1] + target_point_dim, body_fets[0]), nn.ReLU())
        body = [nn.Sequential(nn.Linear(in_fets, out_fets), nn.ReLU())
                for in_fets, out_fets in zip(body_fets, body_fets[1:])]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(body_fets[-1], num_actions)

    def forward(self, ranges, target_points, target_points_activation_map):
        x = [torch.cat((sens, tp.unsqueeze(0),), 1)
             for sens, tp, am in zip(ranges, target_points, target_points_activation_map)]
        x = torch.stack(tuple(x))
        x = self.stump(x)
        x = self.body(x)
        x = self.head(x.view(x.size(0), -1))
        return x


class CartpoleMLPValueEstimator(nn.Module):
    def __init__(self,
                 body_fets=(8, 8, 8),
                 num_actions=3):
        super(CartpoleMLPValueEstimator, self).__init__()
        self.stump = nn.Sequential(nn.Linear(4, body_fets[0]), nn.ReLU())
        body = [nn.Sequential(nn.Linear(in_fets, out_fets), nn.ReLU())
                for in_fets, out_fets in zip(body_fets, body_fets[1:])]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(body_fets[-1], num_actions)

    def forward(self, x):
        x = self.stump(x)
        x = self.body(x)
        x = self.head(x)
        return x
