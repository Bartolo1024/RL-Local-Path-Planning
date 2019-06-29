import torch.nn as nn
import torch.nn.functional as functional


class MLPCartpoleActorCritic(nn.Module):
    def __init__(self,
                 state_shape,
                 body_fets=(64, 32, 16, 8),
                 num_actions=2):
        super(MLPCartpoleActorCritic, self).__init__()
        self.stump = nn.Sequential(nn.Linear(state_shape[0], body_fets[0]), nn.ReLU())
        body = [nn.Sequential(nn.Linear(in_fets, out_fets), nn.ReLU())
                for in_fets, out_fets in zip(body_fets, body_fets[1:])]
        self.body = nn.Sequential(*body)
        self.critic_head = nn.Linear(body_fets[-1], num_actions)
        self.actor_head = nn.Linear(body_fets[-1], num_actions)

    def forward(self, x):
        x = self.stump(x)
        x = self.body(x)
        x = x.view(x.size(0), -1)
        log_probes = functional.softmax(self.actor_head(x), dim=1).clamp(max=1 - 1e-20)
        Q = self.critic_head(x)
        V = (Q * log_probes).sum(1, keepdim=True)
        return log_probes, Q, V
