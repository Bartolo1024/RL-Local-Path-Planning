import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
from functools import wraps

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--session_id', type=str)
    ret = parser.parse_args()
    return ret

def get_rewards(sess_id):
    files = os.listdir("out/{}".format(sess_id))
    reward_files = [file for file in files if file.endswith('Reward_value.txt')]
    reward_names = [file.replace('Reward_value.txt', '') for file in reward_files]
    rewards = dict()
    for file_name, reward_name in zip(reward_files, reward_names):
        with open('out/{}/{}'.format(sess_id, file_name)) as file:
            logs = file.readlines()
            values = [float(log.split(';')[-1].split(':')[-1]) for log in logs]
            rewards[reward_name] = values
    return rewards

def matplotlib_out(width=100, height=10, modes=('save')):
    def _matplotlib_out(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            matplotlib.rcParams['figure.figsize'] = [width, height]
            if 'show' in modes:
                func(*args, **kwargs)
                plt.show()
            if 'save' in modes:
                func(*args, **kwargs)
                plt.savefig('out/{}/{}.png'.format(kwargs['sess_id'], func.__name__))
            plt.close()
        return wrapper
    return _matplotlib_out

@matplotlib_out(width=100, height=10, modes=('save'))
def plot_rewards_hists(rewards, *args, **kwargs):
    for idx, (key, val) in enumerate(rewards.items()):
        plt.subplot(1, len(rewards), idx + 1)
        plt.title(key)
        plt.hist(val)

@matplotlib_out(width=100, height=10, modes=('save'))
def plot_rewards(rewards, smooth_step=1000, *args, **kwargs):
    for idx, (key, val) in enumerate(rewards.items()):
        plt.subplot(1, len(rewards), idx + 1)
        plt.title(key)
        sm_val = [sum(val[beg:end]) / smooth_step
                  for beg, end in zip(range(0, len(val), smooth_step), range(smooth_step, len(val) + 1, smooth_step))]
        plt.plot(sm_val)

def run(session_id):
    rewards = get_rewards(session_id)
    plot_rewards_hists(rewards, sess_id=session_id)
    plot_rewards(rewards, sess_id=session_id)

if __name__ == '__main__':
    args = parse_args()
    run(args.session_id)
