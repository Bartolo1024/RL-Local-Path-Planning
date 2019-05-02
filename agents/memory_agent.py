import torch
from utils import epsilon
import utils.replay_memory
import utils.roll_out_memory
import utils.transitions
import agents.nets as nets
import agents.nets.utils

class MemoryAgent(object):
    def __init__(self,
                 network_architecture,
                 init_state,
                 batch_size=64,
                 gamma=0.99,
                 state_transform=None,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=1000,
                 num_of_actions=3,
                 replay_memory_capacity=10000,
                 sparse_rewards=('HitReward', 'CoinReward'),
                 device=None,
                 recurrent_state_length=5,
                 logger=None,
                 replay_memory='replay'):
        self.step = 0
        self._set_device(device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon.Epsilon(eps_start, eps_end, eps_decay)
        self._set_memory(replay_memory, replay_memory_capacity, sparse_rewards=sparse_rewards)
        self.num_of_actions = num_of_actions
        self.state_transform = state_transform
        if network_architecture in nets.get_recurrent_architectures_list():
            self.state_collector = agents.nets.utils.SequenceCollector(recurrent_state_length, init_state)
        self.network_architecture = network_architecture
        self.logger = logger

    def _set_device(self, device):
        if device is not None and (device is not 'cuda' or torch.cuda.is_available()):
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('MODELS WILL BE TRAINED ON: {} DEVICE'.format(self.device))

    def _set_memory(self, memory, replay_memory_capacity, sparse_rewards=None):
        if memory == 'replay':
            self.memory = utils.replay_memory.ReplayMemory(replay_memory_capacity)
        elif memory == 'rollout' and sparse_rewards is not None:
            self.memory = utils.roll_out_memory.ReplayMemoryWithRollouts(replay_memory_capacity, sparse_rewards)
        else:
            raise NotImplementedError('memory configuration not implemented')

    def sample_memory(self):
        self.epsilon.update()
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        batch = utils.transitions.Transition(*zip(*batch))

        state_batch = self.state_transform(batch.state, self.device)
        action_batch = torch.stack(tuple([torch.tensor(action,
                                                       device=self.device,
                                                       dtype=torch.long,
                                                       requires_grad=False)
                                          for action in batch.action]))
        reward_batch = torch.stack(tuple([torch.tensor(reward,
                                                       device=self.device,
                                                       dtype=torch.float32,
                                                       requires_grad=False)
                                          for reward in batch.reward]))
        return state_batch, action_batch, reward_batch, batch

    def push_transition(self, state, action, next_state, reward, done=False):
        if self.network_architecture in nets.get_recurrent_architectures_list():
            state = self.state_collector(state, actualize=True)
            next_state = self.state_collector(next_state, actualize=False)
        if done:
            self.state_collector.reset()
        self.memory.push(state, action, next_state, reward)
