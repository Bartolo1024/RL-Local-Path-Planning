from reward import Reward

class AccelerationReward(Reward):
    def __init__(self, acc_reward=1.):
        self.acc_reward = acc_reward

    def __call__(self, done, action, *args, **kwargs):
        if not done and action == 0:
            return self.acc_reward
        return 0
