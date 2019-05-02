from torch import nn


class Conv1dAdventageEstimator(nn.Module):
    def __init__(self, state_shape, body_fets=((1, 4), (4, 8), (8, 16), (16, 16))):
        super(Conv1dAdventageEstimator, self).__init__()
        _, w = state_shape
        body = [self._conv_block(in_ch, out_ch)
                for (in_ch, out_ch) in body_fets]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(w * 16, 1)

    def forward(self, x):
        x = self.body(x)
        value = self.head(x)
        return value
