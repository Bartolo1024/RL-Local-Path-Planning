import gym
import args as argparse
from utils import get_new_session_id
from utils import NetSaver
from utils import Logger
from ignite.contrib.handlers import ProgressBar
import ignite.engine
import agents
from environments import GazeboCircuitTurtlebotLidarEnv


class StepGenerator():
    def __init__(self, environment, agent, max_steps):
        self.environment = environment
        self.agent = agent
        self.max_steps = max_steps
        self.reset()

    def __iter__(self):
        return self

    def next(self):
        if self.current > self.max_steps or self.done:
            raise StopIteration
        else:
            action = self.agent.act(self.state)
            prev_state = self.state
            self.state, reward, self.done, _ = self.environment.step(action)
            if self.environment.spec.id == 'CartPole-v0':
                reward = {'Base': reward}
            self.current += 1
            return prev_state, action, self.state, reward

    def __len__(self):
        return self.max_steps

    def reset(self):
        self.state = self.environment.reset()
        self.current = 0
        self.done = False


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
        kwargs = {'port': args.port_ros, 'port_gazebo': args.port_gazebo,
                  'reward_str': args.rewards, 'randomized_target': args.randomized_target}
    else:
        kwargs = {}
    return kwargs


def create_reinforce_engine(agent, environment, args, logger, saver):
    # TODO replace loggr and saver by plugins
    def _run_single_simulation(engine, timestep=None):
        transition = engine.state.batch
        engine.state.agent.push_transition(*transition)

    engine = ignite.engine.Engine(_run_single_simulation)

    @engine.on(ignite.engine.Events.STARTED)
    def initialize(engine):
        engine.state.agent = agent
        engine.state.environment = environment
        engine.state.max_reward = -10000000

    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def _sum_reward(engine):
        transition = engine.state.batch
        _, _, _, reward = transition
        engine.state.total_reward += sum(reward.values())

    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def _update(engine):
        for _ in range(args.num_of_updates):
            loss = engine.state.agent.update()
            logger.log_value('loss', loss)

        if engine.state.epoch % args.target_update == 0 and engine.state.epoch != 0:
            agent.update_target_net()

        logger.update(engine.state.epoch, engine.state.total_reward)

    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def _reset_data(engine):
        engine.state.dataloader.reset()

    @engine.on(ignite.engine.Events.EPOCH_STARTED)
    def _reset(engine):
        engine.state.total_reward = 0

    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def _save_nets(engine):
        if engine.state.epoch != 0 and (engine.state.epoch + 1) % args.save_period == 0:
            saver.save_net(engine.state.epoch + 1, engine.state.total_reward)

        if engine.state.total_reward > engine.state.max_reward:
            engine.state.max_reward = engine.state.total_reward
            saver.save_net(engine.state.epoch + 1, engine.state.total_reward)

    def _attach(plugin):
        plugin.attach(engine)

    engine.attach = _attach

    return engine


def main(args):
    session_id = get_new_session_id()
    logger = Logger(session_id)
    env = gym.make(args.environment_name, **prepare_env_kwargs(args))
    state = env.reset()
    agent = agents.get_agent(args.agent, **prepare_agent_kwargs(args, state, logger, env.action_space.n))
    agent.train()
    saver = NetSaver(agent, args, session_id)

    trainer = create_reinforce_engine(agent, env, args, logger, saver)

    trainer.attach(ProgressBar(persist=False))
    # trainer.attach(logger) # TODO how to write logger

    state = trainer.run(data=StepGenerator(env, agent, max_steps=args.max_steps),
                        max_epochs=args.epochs_count)

    logger.close()
    env.close()


if __name__ == '__main__':
    args = argparse.parse_train_args()
    main(args)
