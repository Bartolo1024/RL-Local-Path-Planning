from reward import Reward

class HitReward(Reward):
    def __init__(self, hit_reward=-1.):
        self.hit_reward = hit_reward

    def __call__(self, done, action, *args, **kwargs):
        if not done:
            reward = 0
        else:
            reward = self.hit_reward
        return reward
