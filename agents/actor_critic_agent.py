import random
import torch
import numpy as np
import torch.optim as optim
import agents.nets as nets
import agents.memory_agent

class A2CAgent(agents.memory_agent.MemoryAgent):
    '''
    Implementation of advantage actor critic (A2C) agent with replay memory
    '''
    def __init__(self,
                 network_architecture,
                 init_state,
                 lr=0.00025,
                 num_of_actions=3,
                 **kwargs):
        super(A2CAgent, self).__init__(network_architecture, init_state, **kwargs)
        self.net = nets.get_actor_critic_net(network_architecture, init_state, num_actions=num_of_actions)
        self.state_transform = nets.get_state_transforms(network_architecture)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.entropy = 0

    def act(self, state):
        if self.network_architecture in nets.get_recurrent_architectures_list():
            state = self.state_collector(state, False)
        state = self.state_transform([state], self.device)
        sample = random.random()
        eps_th = self.epsilon()
        if sample > eps_th:
            with torch.no_grad():
                val, idx = self.net(*state)[0].max(0)
                return idx.item()
        return np.random.randint(self.num_of_actions)

    def update(self):
        tmp = self.sample_memory()
        if tmp is None:
            return
        state_batch, action_batch, reward_batch, batch = tmp
        distribution, _, V = self.net(state_batch)
        log_probes = distribution.log_prob(action_batch)
        entropy = distribution.entropy().mean()
        advantages = self.advantage_estimate(batch, reward_batch, V)

        actor_loss = -(log_probes * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        self.net.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()

    def advantage_estimate(self, batch, reward_batch, values):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = torch.stack(tuple([torch.tensor(s).to(self.device)
                                                   for s in batch.next_state if s is not None]))
        next_state_values = torch.zeros(self.batch_size, device=self.device, requires_grad=False)
        _, _, values = self.net(non_final_next_states)
        next_state_values[non_final_mask] = values
        return reward_batch + next_state_values * self.gamma - values
