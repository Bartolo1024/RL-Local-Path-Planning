import torch
import torch.nn as nn


class GRUValueEstimator(nn.Module):
    def __init__(self,
                 state_shape,
                 hidden_dim=20,
                 num_actions=4):
        super(GRUValueEstimator, self).__init__()
        self.body = nn.GRU(state_shape[1] + 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_actions)
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

    def forward(self, lidar_states_batch, target_points_batch):
        batch_size = lidar_states_batch.shape[1]
        lidar_states_batch = lidar_states_batch.squeeze(2)
        x = [torch.cat((sens, tp), 1) for sens, tp in zip(lidar_states_batch, target_points_batch)]
        x = torch.stack(tuple(x))
        hidden = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
        x, _ = self.body(x, hidden)
        x = x.view((-1, batch_size, self.hidden_dim))[-1]
        out = self.head(x)
        return out
