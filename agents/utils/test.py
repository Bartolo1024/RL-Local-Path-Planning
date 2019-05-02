import roll_out_memory

if __name__ == '__main__':
    '''
    out:
    PathReward:      1      AccReward:       0      CoinReward:      1      HitReward:       0 
    PathReward:      1      AccReward:       0      CoinReward:      1      HitReward:       0.64 
    PathReward:      0      AccReward:       0.512  CoinReward:      0.8    HitReward:       0.8 
    PathReward:      0.512  AccReward:       0.64   CoinReward:      1      HitReward:       1 
    PathReward:      0.64   AccReward:       0.8    CoinReward:      0      HitReward:       0 
    PathReward:      0.8    AccReward:       1      CoinReward:      0      HitReward:       0 
    PathReward:      1      AccReward:       1      CoinReward:      0.512  HitReward:       0 
    PathReward:      0.64   AccReward:       0      CoinReward:      0.64   HitReward:       0 
    PathReward:      0.8    AccReward:       0      CoinReward:      0.8    HitReward:       0
    '''
    mem = roll_out_memory.ReplayMemoryWithRollouts(9, ('HitReward', 'AccReward', 'PathReward', 'CoinReward'), max_roll_out_length=3, gamma=0.8)
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 0, 'CoinReward': 0, 'PathReward': 0})
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 0, 'CoinReward': 1, 'PathReward': 1})
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 0, 'CoinReward': 0, 'PathReward': 0})
    mem.push(1, 2, 3, {'HitReward': 1, 'AccReward': 0, 'CoinReward': 1, 'PathReward': 0})
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 0, 'CoinReward': 0, 'PathReward': 0})
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 1, 'CoinReward': 0, 'PathReward': 0})
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 1, 'CoinReward': 0, 'PathReward': 1})
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 0, 'CoinReward': 0, 'PathReward': 0})
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 0, 'CoinReward': 0, 'PathReward': 0})
    mem.push(1, 2, 3, {'HitReward': 0, 'AccReward': 0, 'CoinReward': 1, 'PathReward': 1})
    for el in mem.memory:
        print(''.join(['{}: \t {} \t'.format(key, val) for key, val in el.reward.dict.items()]))
