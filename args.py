import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_train_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--environment-name', type=str, default='myenv-v0',
                        choices=('myenv-v0', 'CartPole-v0', 'env-maze-v0'),
                        help='It is an environment name registered in gym. '
                             'Cart pole can be used to low computational cost algorithms testing')
    parser.add_argument('--rewards', type=str, default='PathReward,HitReward')
    parser.add_argument('-a', '--agent', type=str, default='dqn', choices=('dqn', 'a2c'))
    parser.add_argument('-net', '--network-architecture', type=str, default='mlp',
                        help='architecture of network')
    parser.add_argument('-e', '--epochs-count', type=int, default=20000,
                        help='epochs count')
    parser.add_argument('-t', '--target_update', type=int, default=200,
                        help='epochs between target network updates')
    parser.add_argument('-es', '--eps-start', type=float, default=0.9,
                        help='random action start probability')
    parser.add_argument('-ee', '--eps-end', type=float, default=0.05,
                        help='random action end probability')
    parser.add_argument('-ed', '--eps-decay', type=int, default=1000,
                        help='random action probability decay')
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('-nu', '--num-of-updates', type=int, default=1,
                        help='number of updates after epoch '
                             '(useful when gpu capacity is low)')
    parser.add_argument('-g', '--gamma', type=float, default=0.99,
                        help='gamma')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.00025,
                        help='learning rate')
    parser.add_argument('--no-render', type=bool, help='display game')
    parser.add_argument('--set-path-controller', type=bool,
                        help='add target point to state')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='max steps count')
    parser.add_argument('--memory-capacity', type=int, default=10000,
                        help='replay memory capacity')
    parser.add_argument('--save-period', type=int, default=1000,
                        help='epochs between net parameters stores')
    parser.add_argument('--recurrent-state-length', type=int, default=5,
                        help='collected time steps in case of usage recurrent nn architecture')
    parser.add_argument('-mem', '--memory', type=str, default='replay',
                        choices=('replay', 'rollout'))
    parser.add_argument('-sr', '--sparse-rewards', nargs='+', default=('HitReward',))
    parser.add_argument('-pg', '--port-gazebo', type=str, default='11345')
    parser.add_argument('-pr', '--port-ros', type=str, default='11311')
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--randomized_target', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Randomized path.")
    ret = parser.parse_args()
    return ret


def parse_eval_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--environment-name', type=str, default='myenv-v0',
                        choices=('myenv-v0', 'CartPole-v0', 'env-maze-v0'),
                        help='It is an environment name registered in gym. '
                             'Cart pole can be used to low computational cost algorithms testing')
    parser.add_argument('-a', '--agent', type=str, default='dqn', choices=('dqn', 'a2c', 'dwa', 'simple_dwa'))
    parser.add_argument('--rewards', type=str, default='CoinReward')
    parser.add_argument('--weights', type=str, help='weights path', default=None)
    parser.add_argument('-v', '--value-estimator', type=str, default='mlp',
                        help='architecture of value network')
    parser.add_argument('-pg', '--port-gazebo', type=str, default='11345')
    parser.add_argument('-pr', '--port-ros', type=str, default='11311')
    ret = parser.parse_args()
    return ret


def parse_test_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--rewards', type=str, default='CoinReward')
    parser.add_argument('--environment-name', type=str, default='myenv-v0')
    parser.add_argument('--mode', choices=('test_gym', 'test_planner', 'test_cords'),
                        default='test_gym')
    ret = parser.parse_args()
    return ret


def prepare_env_kwargs(args):
    if args.environment_name in ('myenv-v0', 'env-maze-v0'):
        kwargs = {
            'port': args.port_ros,
            'port_gazebo': args.port_gazebo,
            'reward_str': args.rewards
        }
    else:
        kwargs = {}
    return kwargs