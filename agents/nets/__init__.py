from agents.nets.value_estimators.conv1d_value_estimator import Conv1dValueEstimator
from agents.nets.value_estimators.conv2d_value_estimator import Conv2dValueEstimator
from agents.nets.value_estimators.lstm_value_estimator import LSTMValueEstimator
from agents.nets.value_estimators.mlp_value_estimator import MLPValueEstimator
# import agents.nets.policy.conv1d_policy as conv1d_policy
# import agents.nets.adventage_estimators.conv1d_adventage_estimator as conv1d_adventage_estimator

from .utils import SequenceCollector
import transforms

def get_value_estimator(name,
                        state_shape,
                        body_fets=(1, 4, 8, 16, 16),
                        num_actions=4):
    body_fets = ((f1, f2) for (f1, f2) in zip(body_fets, body_fets[1:]))
    if name == 'conv1d':
        return Conv1dValueEstimator(state_shape, body_fets, num_actions)
    elif name == 'conv2d':
        return Conv2dValueEstimator(state_shape, body_fets, num_actions)
    elif name == 'lstm':
        return LSTMValueEstimator(state_shape, num_actions=num_actions)
    elif name == 'mlp':
        return MLPValueEstimator(state_shape, body_fets, num_actions)
    else:
        assert 'BAD VALUE ESTIMATOR NAME'

def get_adventage_estimator(name,
                            state_shape,
                            num_actions=4):
    if name is 'conv1d':
        return conv1d_adventage_estimator.Conv1dAdventageEstimator(state_shape, num_actions)
    raise NotImplementedError('BAD VALUE ESTIMATOR NAME')

def get_policy(name,
               state_shape,
               num_actions=4):
    if name is 'conv1d':
        return conv1d_policy.Conv1dPolicy(state_shape, num_actions)
    raise NotImplementedError('BAD VALUE ESTIMATOR NAME')

def get_state_transforms(net_architecture, **kwargs):
    if net_architecture == 'conv1d':
        return transforms.ToTensor()
    elif net_architecture == 'lstm':
        return transforms.ToRecurrentStatesTensor()
    return NotImplementedError

def get_recurrent_architectures_list():
    return ['lstm', 'rnn', 'gru', 'transformer']
