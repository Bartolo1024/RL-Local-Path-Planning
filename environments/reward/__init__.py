from .hit_reward import HitReward
from .coin_reward import CoinReward
from .path_reward import PathReward
from .acceleration_reward import AccelerationReward

def get_reward(name, config, path_planner):
    rewards = name.split(',')
    ret = []
    if 'HitReward' in rewards:
        ret.append(HitReward())
    if 'PathReward' in rewards:
        ret.append(PathReward(config=config, path_planner=path_planner))
    if 'CoinReward' in rewards:
        ret.append(CoinReward(config=config, path_planner=path_planner))
    if 'AccelerationReward' in rewards:
        ret.append(AccelerationReward())
    return ret
