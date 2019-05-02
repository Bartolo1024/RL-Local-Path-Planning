from torch import nn
from collections import deque

def conv1d_block(in_feats, out_feats):
    conv = nn.Conv1d(in_feats, out_feats, kernel_size=5, stride=1, padding=2)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, relu)

def _conv2d_block(in_feats, out_feats):
    conv = nn.Conv2d(in_feats, out_feats, kernel_size=5, stride=2, padding=2)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, relu)

class SequenceCollector(object):
    def __init__(self, state_length, init_state):
        self.state_length = state_length
        self.states_queue = deque()
        self.init_state = init_state
        self.reset()

    def __call__(self, state, actualize):
        if actualize:
            self.states_queue.append(state)
            if len(self.states_queue) > self.state_length:
                self.states_queue.popleft()
            return list(self.states_queue)
        else:
            return [state] + list(self.states_queue)

    def reset(self):
        self.states_queue.clear()
        for _ in range(self.state_length):
            self.__call__(self.init_state, True)
