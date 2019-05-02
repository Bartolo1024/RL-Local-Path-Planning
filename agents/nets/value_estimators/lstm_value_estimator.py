import torch
import torch.nn as nn
from agents.nets.utils import conv1d_block

class LSTMValueEstimator(nn.Module):
    def __init__(self,
                 state_shape,
                 stump_fets=(32, 16, 8),
                 num_actions=4):
        super(LSTMValueEstimator, self).__init__()
        _, w = state_shape
        stump_fets = [1] + list(stump_fets)
        body = [conv1d_block(in_ch, out_ch)
                for in_ch, out_ch in zip(stump_fets, stump_fets[1:])]
        self.body = nn.Sequential(*body)
        self.lstm_stump = nn.Sequential(nn.Linear(stump_fets[-1] * w + 2, stump_fets[-1]), nn.ReLU())
        self.lstm = nn.LSTM(stump_fets[-1], num_actions)
        self.num_actions = num_actions

    def forward(self, lidar_states_batch, target_points_batch):
        batch_size = lidar_states_batch.shape[1]
        lidar_features = [self.body(inp).view(batch_size, -1) for inp in lidar_states_batch]
        x = [torch.cat((sens, tp), 1) for sens, tp in zip(lidar_features, target_points_batch)]
        x = [self.lstm_stump(inp) for inp in x]
        x = torch.stack(tuple(x))
        hidden = (torch.zeros(1, batch_size, self.num_actions, device=x.device),
                  torch.zeros(1, batch_size, self.num_actions, device=x.device))
        out, _ = self.lstm(x, hidden)
        return out.view((-1, batch_size, self.num_actions))[-1]
