import argparse
import gym
import cv2
from agents.dqn_agent import DQNAgent
from environments import GazeboCircuitTurtlebotLidarEnv # must be imported to register in gym

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--rewards', type=str, default='CoinReward')
    parser.add_argument('--environment-name', type=str, default='myenv-v0')
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('-v', '--value-estimator', type=str, default='conv1d',
                        help='architecture of value network')
    ret = parser.parse_args()
    return ret

def step_generator(env, agent):
    state, done = env.reset()
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        yield state, action, next_state, reward
        state = next_state

def main(args):
    env = gym.make(args.environment_name)
    env.set_rewards(args.rewards)
    state, _ = env.reset()
    agent = DQNAgent(args.value_estimator,
                     init_state = state,
                     num_of_actions=3) # F,L,R
    agent.load_weights(args.weights)

    total_reward = 0

    end = False
    while not end:

        for state, action, next_state, reward in step_generator(env, agent):
            total_reward += reward

        print('total reward: {}'.format(total_reward))

        key = cv2.waitKey(0)
        if key == ord('q'):
            end = True
        else:
            total_reward = 0

    env.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)