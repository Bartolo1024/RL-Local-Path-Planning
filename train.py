import argparse
import gym
from utils import get_new_session_id
from utils import NetSaver
from utils import Logger
from agents.dqn_agent import DQNAgent
from environments import GazeboCircuitTurtlebotLidarEnv

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--environment-name', type=str, default='myenv-v0')
    parser.add_argument('--rewards', type=str, default='PathReward,HitReward')
    parser.add_argument('-a', '--agent', type=str, default='dqn', choices=('dqn', 'a2c'))
    parser.add_argument('-net', '--network-architecture', type=str, default='conv1d',
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
    parser.add_argument('-sr', '--sparse-rewards', type=tuple, default=('HitReward',))
    ret = parser.parse_args()
    return ret

def step_generator(env, agent, max_steps=1000):
    state, done = env.reset()
    for _ in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        yield state, action, next_state, reward
        state = next_state
        if done:
            break

def main(args):
    max_reward = -100000
    session_id = get_new_session_id()
    env = gym.make(args.environment_name)
    env.set_rewards(args.rewards)
    state, _ = env.reset()
    logger = Logger(session_id)
    agent = DQNAgent(args.network_architecture,
                     init_state=state, #state['ranges'].shape,
                     lr=args.learning_rate,
                     eps_start=args.eps_start,
                     eps_end=args.eps_end,
                     eps_decay=args.eps_decay,
                     batch_size=args.batch_size,
                     num_of_actions=3,
                     recurrent_state_length=args.recurrent_state_length,
                     replay_memory=args.memory,
                     replay_memory_capacity=args.memory_capacity,
                     sparse_rewards=args.sparse_rewards,
                     logger=logger)
    saver = NetSaver(agent, args, session_id)
    agent.logger = logger
    env.set_logger(logger)

    for i_episode in range(args.epochs_count):

        total_reward = 0

        for state, action, next_state, reward in step_generator(env, agent, args.max_steps):

            agent.push_transition(state, action, next_state, reward)

            total_reward += sum(reward.values())

        logger.update(i_episode, total_reward)

        for _ in range(args.num_of_updates):
            agent.update()

        if i_episode % args.target_update == 0 and i_episode != 0:
            agent.update_target_net()

        if i_episode != 0 and (i_episode + 1) % args.save_period == 0:
            saver.save_net(i_episode + 1, total_reward)

        if total_reward > max_reward:
            max_reward = total_reward
            saver.save_net(i_episode + 1, total_reward)

    logger.close()
    env.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
