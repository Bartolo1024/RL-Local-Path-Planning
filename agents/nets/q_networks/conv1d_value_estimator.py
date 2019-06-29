import torch
import torch.nn as nn

class Conv1dValueEstimator(nn.Module):
    def __init__(self,
                 state_shape,
                 body_fets,
                 num_actions=4):
        super(Conv1dValueEstimator, self).__init__()
        _, w = state_shape
        w = w + 2
        body = [self._conv_block(in_ch, out_ch)
                for (in_ch, out_ch) in body_fets]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(w * 16, num_actions)

    @staticmethod
    def _conv_block(in_feats, out_feats):
        conv = nn.Conv1d(in_feats, out_feats, kernel_size=5, stride=1, padding=2)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, relu)

    def forward(self, ranges, target_points):
        x = [torch.cat((sens, tp.unsqueeze(0)), 1) for sens, tp in zip(ranges, target_points)]
        x = torch.stack(tuple(x))
        x = self.body(x)
        x = self.head(x.view(x.size(0), -1))
        return x
    