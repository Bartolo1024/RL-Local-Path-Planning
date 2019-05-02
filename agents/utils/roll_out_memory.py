import random
import transitions
import replay_memory


class Reward(object):
    def __init__(self, rewards_dict):
        self.dict = rewards_dict

    @property
    def value(self):
        return sum(self.dict.values())

    def __getitem__(self, item):
        return self.dict[item]

    def __setitem__(self, key, value):
        self.dict[key] = value


class ReplayMemoryWithRollouts(replay_memory.ReplayMemory):
    def __init__(self, capacity, sparse_rewards, max_roll_out_length=5, gamma=.5):
        super(ReplayMemoryWithRollouts, self).__init__(capacity)
        self.sparse_rewards = sparse_rewards
        self.max_roll_out_length = max_roll_out_length
        self.gamma = gamma

    def push(self, state, action, next_state, reward):
        transition = transitions.Transition(state, action, next_state, Reward(reward))
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        self.memory[self.position] = transition
        self._roll_out_rewards()
        self.position = (self.position + 1) % self.capacity

    def _roll_out_rewards(self):
        for rew in self.sparse_rewards:
            rolled_reward = self.memory[self.position].reward[rew]
            if rolled_reward == 0:
                continue
            for idx in range(self.max_roll_out_length):
                pos = (self.position - idx - 1) % self.capacity
                if pos + 1 > len(self.memory):
                    break
                if self.memory[pos].reward[rew] == 0:
                    self.memory[pos].reward[rew] = rolled_reward * self.gamma ** (idx + 1)
                else:
                    break

    def sample(self, batch_size):
        ret = random.sample(self.memory, batch_size)
        return [transitions.Transition(rew.state, rew.action, rew.next_state, rew.reward.value)
                for rew in ret]
