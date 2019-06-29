import args as argparse
import gym
import cv2
import tests.path_planner
import tests.transform_cords
import agents.human_agent
from environments import GazeboCircuitTurtlebotLidarEnv # must be imported to register in gym


def step_generator(env, agent):
    state, done = env.reset()
    while not done:
        action = agent.act(state)
        print(action)
        next_state, reward, done, _ = env.step(action)
        yield state, action, next_state, reward
        state = next_state


def environment_test(args):
    env = gym.make(args.environment_name)
    env.set_rewards(args.rewards)
    state, _ = env.reset()
    agent = agents.human_agent.HumanAgent()
    total_reward = 0

    end = False
    while not end:

        for state, action, next_state, reward in step_generator(env, agent):
            print(reward)
            total_reward += sum(reward.values())

        print('total reward: {}'.format(total_reward))
        key = cv2.waitKey(0)
        if key == ord('q'):
            end = True
        else:
            total_reward = 0

    env.close()
    cv2.destroyWindow('steering window')


if __name__ == '__main__':
    args = argparse.parse_test_args()
    if args.mode == 'test_gym':
        environment_test(args)
    elif args.mode == 'test_planner':
        tests.path_planner.run_path_planner_test()
    elif args.mode == 'test_cords':
        tests.transform_cords.run_transform_test()
    else:
        raise NotImplementedError
