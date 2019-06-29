import torch
import torch.nn as nn
import torch.nn.functional as functional


class MLPActorCritic(nn.Module):
    def __init__(self,
                 ranges_shape,
                 target_points_shape,
                 body_fets=(64, 32, 16, 8),
                 num_actions=3):
        super(MLPActorCritic, self).__init__()
        self.stump = nn.Sequential(nn.Linear(ranges_shape[1] + target_points_shape[0], body_fets[0]), nn.ReLU())
        body = [nn.Sequential(nn.Linear(in_fets, out_fets), nn.ReLU())
                for in_fets, out_fets in zip(body_fets, body_fets[1:])]
        self.body = nn.Sequential(*body)
        self.critic_head = nn.Linear(body_fets[-1], num_actions)
        self.actor_head = nn.Linear(body_fets[-1], num_actions)

    def forward(self, ranges, target_points):
        x = [torch.cat((sens, tp.unsqueeze(0)), 1) for sens, tp in zip(ranges, target_points)]
        x = torch.stack(tuple(x))
        x = self.stump(x)
        x = self.body(x)
        x = x.view(x.size(0), -1)
        log_probes = functional.softmax(self.actor_head(x), dim=1).clamp(max=1 - 1e-20)
        Q = self.critic_head(x)
        V = (Q * log_probes).sum(1, keepdim=True)
        return log_probes, Q, V
