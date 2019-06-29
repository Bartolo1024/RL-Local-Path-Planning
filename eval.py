import args as argparse
import gym
import cv2
import agents
from utils import path_plot
from environments import GazeboCircuitTurtlebotLidarEnv # must be imported to register in gym
import pathlib2

def step_generator(env, agent):
    state = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if env.spec.id == 'CartPole-v0':
            reward = {'Base': reward}
        yield state, action, next_state, reward
        state = next_state
        if done:
            break


def main(args):
    env = gym.make(args.environment_name, **argparse.prepare_env_kwargs(args))
    state = env.reset()
    agent = agents.get_agent(args.agent, env_name=args.environment_name,
                             network_architecture=args.value_estimator,
                             init_state=state, num_of_actions=3)
    if args.agent in ['dqn', 'a2c']:
        out_path = pathlib2.Path('/'.join(args.weights.split('/')[:-1]))
        agent.load_weights(args.weights)
        agent.eval()
    else:
        out_path = pathlib2.Path('out/{}'.format(args.agent))
    total_reward = 0

    path_plotter = path_plot.Plotter(args.environment_name, out_path)

    end = False
    while not end:

        for state, action, next_state, reward in step_generator(env, agent):
            total_reward += sum(reward.values())
            path_plotter.update(state, reward)

        print('total reward: {}'.format(total_reward))
        path_plotter.reset()

        key = cv2.waitKey(0)
        if key == ord('q'):
            end = True
        else:
            total_reward = 0

    env.close()


if __name__ == '__main__':
    args = argparse.parse_eval_args()
    main(args)
