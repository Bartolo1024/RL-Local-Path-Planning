import random
import numpy as np
import torch
from torch import optim
from torch.nn import functional
import memory_agent
import agents.nets as nets


class DQNAgent(memory_agent.MemoryAgent):
    def __init__(self,
                 network_architecture,
                 init_state,
                 lr=0.00025,
                 num_of_actions=3,
                 pretrained=None,
                 **kwargs):
        super(DQNAgent, self).__init__(network_architecture, init_state, **kwargs)
        self.qnet = nets.get_value_estimator(network_architecture,
                                             init_state,
                                             num_actions=num_of_actions,
                                             pretrained=pretrained).to(self.device)
        self.qnet = torch.nn.DataParallel(self.qnet, device_ids=range(torch.cuda.device_count()))
        self.target_net = nets.get_value_estimator(network_architecture,
                                                   init_state,
                                                   num_actions=num_of_actions,
                                                   pretrained=pretrained).to(self.device)
        self.target_net = torch.nn.DataParallel(self.target_net, device_ids=range(torch.cuda.device_count()))
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.qnet.parameters(), lr=lr)
        self.state_transform = nets.get_state_transforms(network_architecture)
        self.num_of_actions = num_of_actions

    def act(self, state):
        if self.network_architecture in nets.get_recurrent_architectures_list():
            state = self.state_collector(state, False)
        state = self.state_transform([state], self.device)
        sample = random.random()
        eps_th = self.epsilon()
        if sample > eps_th or self.eval_mode:
            with torch.no_grad():
                val, idx = self.qnet(*state)[0].max(0)
                return idx.item()
        return np.random.randint(self.num_of_actions)

    def update(self):
        tmp = self.sample_memory()
        if tmp is None:
            print('not update')
            return
        state_batch, action_batch, reward_batch, batch = tmp
        self.qnet.train()
        q = self.qnet(*state_batch)

        state_action_values = q.gather(1, action_batch.view(self.batch_size, 1))

        with torch.no_grad():
            expected_state_action_values = self.future_reward_estimate(batch) + reward_batch

        loss = functional.l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.logger.log_value('loss', loss.item())
        # print('loss: {}'.format(loss.item()))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnet.parameters():
            if param.requires_grad:
                param.grad.data.clamp(-1, 1)
        self.optimizer.step()
        return loss.item()

    def future_reward_estimate(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = filter(lambda el: el is not None, batch.next_state)
        non_final_next_states = self.state_transform(non_final_next_states, self.device)
        next_state_values = torch.zeros(self.batch_size, device=self.device, requires_grad=False)
        next_state_values[non_final_mask] = self.target_net(*non_final_next_states).max(1)[0].detach()
        return next_state_values * self.gamma

    def update_target_net(self):
        self.target_net.load_state_dict(self.qnet.state_dict())

    def save_qnet(self, dir):
        torch.save(self.qnet.state_dict(), dir)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.qnet.load_state_dict(state_dict)
