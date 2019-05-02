import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from .nets.adventage_estimators import Conv1dAdventageEstimator as Critic
from .nets.policy import Conv1dPolicy as Actor
from .nets import get_state_transforms
from .memory_agent import MemoryAgent
import agents.nets as nets

class A2CAgent(MemoryAgent):
    '''
    Implementation of classic advantage actor critic (A2C) agent with replay memory
    '''
    def __init__(self,
                 network_architecture,
                 init_state,
                 lr=0.00025,
                 num_of_actions=3,
                 **kwargs):
        super(self, ACAgent).__init__(**kwargs)
        self.actor = Actor(init_state)
        self.critic = Critic(init_state, num_of_actions)
        self.actor = nets.get_adventage_estimator(network_architecture)
        self.actor = nets.get_policy(network_architecture)
        self.state_transform = nets.get_state_transforms(network_architecture)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.entropy = 0

    def act(self, state):
        if self.network_architecture in nets.get_recurrent_architectures_list():
            state = self.state_collector(state, False)
        state = self.state_transform([state], self.device)
        sample = random.random()
        eps_th = self.epsilon()
        if sample > eps_th:
            with torch.no_grad():
                val, idx = self.actor(*state)[0].max(0)
                return idx.item()
        return np.random.randint(self.num_of_actions)

    def update(self):
        tmp = self.sample_memory()
        if tmp is None:
            return
        state_batch, action_batch, reward_batch, batch = tmp
        distributions = self.actor(state_batch)

        # update critic
        self.critic.zero_grad()
        values = self.critic(state_batch)
        expected_values = reward_batch + self.future_value_estimate(batch)
        critic_loss = F.smooth_l1_loss(values, expected_values)
        critic_loss.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp(-1, 1)
        self.critic_optimizer.step()

        # update actor
        log_probs = distributions.gather(1, action_batch.view(self.batch_size, 1))
        # TODO entropy loss
        adventages = reward_batch - values
        policy_gradient_loss = - torch.mean(log_probs * adventages)
        actor_loss = policy_gradient_loss # TODO add entropy
        actor_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp(-1, 1)
        self.actor_optimizer.step()

    def future_value_estimate(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = torch.stack(tuple([torch.tensor(s).to(self.device)
                                                   for s in batch.next_state if s is not None]))
        next_state_values = torch.zeros(self.batch_size, device=self.device, requires_grad=False)
        next_state_values[non_final_mask] = self.critic(non_final_next_states)
        return next_state_values * self.gamma
