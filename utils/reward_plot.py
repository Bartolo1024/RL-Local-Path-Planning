import matplotlib
import matplotlib.pyplot as plt
from utils import plugin


class RewardPlotter(plugin.BasePlugin):
    def __init__(self, env_name, out_path):
        super(RewardPlotter, self).__init__(env_name, out_path)
        self.rewards_per_epoch = {}

    def update(self, state, reward):
        for key, val in reward.items():
            prev_val = reward.get(key, None)
            if prev_val:
                self.rewards_per_epoch[key] = [val]
            else:
                self.rewards_per_epoch[key].append(val)
        self.rewards_per_epoch['time'] = state['time']

    def reset(self):
        sum_overall = 0
        for key, value in self.rewards_per_epoch.items():
            sum_one = sum(value)
            sum_overall += sum_one
        self.rewards_per_epoch = {}

    def plot_rewards_histogram(self, rewards):
        pass

    def plot_rewards_hists(self, rewards, width=100, height=10):
        matplotlib.rcParams['figure.figsize'] = [width, height]
        for idx, (key, val) in enumerate(rewards.items()):
            plt.subplot(1, len(rewards), idx + 1)
            plt.title(key)
            plt.hist(val)
            plt.savefig(str(self.out_path / 'hist.png'))

    def plot_reward(self, reward, smooth_step=1000, width=100, height=10):
        matplotlib.rcParams['figure.figsize'] = [width, height]
        plt.subplot(1, len(rewards), idx + 1)
        plt.title(key)
        sm_val = [sum(val[beg:end]) / smooth_step
                  for beg, end in zip(range(0, len(val), smooth_step), range(smooth_step, len(val) + 1, smooth_step))]
        plt.plot(sm_val)
        plt.savefig(str(self.out_path / 'hist.png'))