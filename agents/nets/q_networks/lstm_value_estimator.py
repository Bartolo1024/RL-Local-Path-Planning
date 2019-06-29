import torch
import torch.nn as nn
import itertools


class LSTMValueEstimator(nn.Module):
    def __init__(self,
                 state_shape,
                 stump_fets=(64, 32, 16, 8),
                 num_actions=4):
        super(LSTMValueEstimator, self).__init__()
        self.stump = nn.Sequential(nn.Linear(state_shape[1], stump_fets[0]), nn.ReLU())
        body = [nn.Sequential(nn.Linear(in_fets, out_fets), nn.ReLU())
                for in_fets, out_fets in zip(stump_fets, stump_fets[1:])]
        self.body = nn.Sequential(*body)
        self.lstm = nn.LSTM(stump_fets[-1] + 2, num_actions)
        self.num_actions = num_actions

    def forward(self, lidar_states_batch, target_points_batch):
        batch_size = lidar_states_batch.shape[1]
        lidar_features = [self.stump(inp).view(batch_size, -1) for inp in lidar_states_batch]
        lidar_features = [self.body(inp).view(batch_size, -1) for inp in lidar_features]
        x = [torch.cat((sens, tp), 1) for sens, tp in zip(lidar_features, target_points_batch)]
        x = torch.stack(tuple(x))
        hidden = (torch.zeros(1, batch_size, self.num_actions, device=x.device),
                  torch.zeros(1, batch_size, self.num_actions, device=x.device))
        out, _ = self.lstm(x, hidden)
        return out.view((-1, batch_size, self.num_actions))[-1]

    def load_pretrained_weights(self, path, freeze):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)
        if freeze:
            for param in itertools.chain(self.stump.parameters(), self.body.parameters()):
                param.requires_grad = False
