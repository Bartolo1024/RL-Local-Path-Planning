import torch
from agents.nets.q_networks.conv1d_value_estimator import Conv1dValueEstimator
from agents.nets.q_networks.conv2d_value_estimator import Conv2dValueEstimator
from agents.nets.q_networks.lstm_value_estimator import LSTMValueEstimator
from agents.nets.q_networks.mlp_value_estimator import MLPValueEstimator, CartpoleMLPValueEstimator
from agents.nets.q_networks.gru import GRUValueEstimator
from agents.nets.ac_models.mlp_model import MLPActorCritic
from agents.nets.ac_models.mlp_cartpole_model import MLPCartpoleActorCritic

from .utils import SequenceCollector
from . import transforms

def get_value_estimator(name,
                        init_state,
                        pretrained=None,
                        freeze=True,
                        body_fets=(1, 4, 8, 16, 16),
                        num_actions=3):
    body_fets = ((f1, f2) for (f1, f2) in zip(body_fets, body_fets[1:]))
    if name == 'conv1d':
        return Conv1dValueEstimator(init_state['ranges'].shape, body_fets, num_actions)
    elif name == 'conv2d':
        return Conv2dValueEstimator(init_state['ranges'].shape, body_fets, num_actions)
    elif name == 'lstm':
        net = LSTMValueEstimator(init_state['ranges'].shape, num_actions=num_actions)
        if pretrained:
            net.load_pretrained_weights(pretrained, freeze)
        return net
    elif name == 'mlp':
        return MLPValueEstimator(init_state['ranges'].shape, num_actions=num_actions)
    elif name == 'gru':
        return GRUValueEstimator(init_state['ranges'].shape, num_actions=num_actions)
    elif name == 'cartpole_mlp':
        return CartpoleMLPValueEstimator(num_actions=num_actions)
    else:
        assert 'BAD VALUE ESTIMATOR NAME'

def get_actor_critic_net(name,
                         init_state,
                         num_actions=3,
                         env='cartpole'):
    if name == 'mlp':
        return MLPActorCritic(ranges_shape=init_state['ranges'].shape,
                              target_points_shape=init_state['target_point'].shape,
                              num_actions=num_actions)
    elif name == 'cartpole_mlp' or (env == 'cartpole' and name == 'mlp'):
        return MLPCartpoleActorCritic(state_shape=init_state.shape,
                                      num_actions=num_actions)
    raise NotImplementedError('BAD VALUE ESTIMATOR NAME')

def get_state_transforms(net_architecture, **kwargs):
    if net_architecture in ('conv1d', 'mlp'):
        return transforms.ToTensor()
    elif net_architecture in ('cartpole_mlp',):
        return to_tensor
    elif net_architecture in get_recurrent_architectures_list():
        return transforms.ToRecurrentStatesTensor()
    return NotImplementedError

def to_tensor(state, device):
    state = list(state)
    return (torch.tensor(state, device=device, dtype=torch.float32, requires_grad=False),)

def get_recurrent_architectures_list():
    return ['lstm', 'rnn', 'gru', 'transformer']
