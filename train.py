import args as argparse
import gym
from utils import get_new_session_id
from utils import NetSaver
from utils import Logger
import agents
from environments import GazeboCircuitTurtlebotLidarEnv


def step_generator(env, agent, max_steps=1000):
    state = env.reset()
    for _ in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if env.spec.id == 'CartPole-v0':
            reward = {'Base': reward}
        yield state, action, next_state, reward
        state = next_state
        if done:
            break


def prepare_agent_kwargs(args, state, logger, num_of_actions):
    agent_kwargs = dict(network_architecture=args.network_architecture,
                        init_state=state,
                        lr=args.learning_rate,
                        eps_start=args.eps_start,
                        eps_end=args.eps_end,
                        eps_decay=args.eps_decay,
                        batch_size=args.batch_size,
                        num_of_actions=num_of_actions,
                        recurrent_state_length=args.recurrent_state_length,
                        replay_memory=args.memory,
                        replay_memory_capacity=args.memory_capacity,
                        sparse_rewards=tuple(args.sparse_rewards),
                        pretrained=args.pretrained,
                        logger=logger)
    return agent_kwargs


def prepare_env_kwargs(args):
    if args.environment_name in ('myenv-v0',):
        kwargs = {'port': args.port_ros, 'port_gazebo': args.port_gazebo, 'reward_str': args.rewards}
    else:
        kwargs = {}
    return kwargs


def main(args):
    max_reward = -100000
    session_id = get_new_session_id()
    logger = Logger(session_id)
    env = gym.make(args.environment_name, **prepare_env_kwargs(args))
    state = env.reset()
    agent = agents.get_agent(args.agent, **prepare_agent_kwargs(args, state, logger, env.action_space.n))
    agent.train()
    saver = NetSaver(agent, args, session_id)

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
    args = argparse.parse_train_args()
    main(args)
