import re
import gym
import sys
import args as argparse
import agents
import utils.ignite_engine as ignite_engine # TODO replace with ignite.engine if not debug version
from utils import get_new_session_id
from utils import NetSaver
from utils import Logger
from ignite.contrib.handlers import ProgressBar
from utils.step_generator import MultienvStepGenerator
from environments import GazeboCircuitTurtlebotLidarEnv


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


def create_reinforce_engine(agent, environments, args):
    def _run_single_simulation(engine, timestep=None):
        transition = engine.state.batch
        engine.state.agent.push_transition(*transition)

    engine = ignite_engine.Engine(_run_single_simulation)

    @engine.on(ignite_engine.Events.STARTED)
    def initialize(engine):
        engine.state.agent = agent
        engine.state.environment = environments
        engine.state.max_reward = -sys.maxint

    @engine.on(ignite_engine.Events.EPOCH_STARTED)
    def _reset(engine):
        engine.state.total_reward = 0
        engine.state.loss = 0.

    @engine.on(ignite_engine.Events.ITERATION_COMPLETED)
    def _sum_reward(engine):
        transition = engine.state.batch
        _, _, _, reward = transition
        engine.state.total_reward += sum(reward.values())

    @engine.on(ignite_engine.Events.EPOCH_COMPLETED)
    def _update(engine):
        for _ in range(args.num_of_updates):
            loss = engine.state.agent.update()
            engine.state.loss += loss

        if engine.state.epoch % args.target_update == 0 and engine.state.epoch != 0:
            agent.update_target_net()

    @engine.on(ignite_engine.Events.EPOCH_COMPLETED)
    def _reset_data(engine):
        engine.state.dataloader.reset()

    @engine.on(ignite_engine.Events.COMPLETED)
    def close(engine):
        [env.close() for env in environments]

    def _attach(plugin):
        plugin.attach(engine)

    engine.attach = _attach

    return engine


def main(args):
    session_id = get_new_session_id()
    logger = Logger(session_id)

    env_names = re.findall(r'[\(|,]([^\(|,|\)]+)', args.environment_name)
    envs = []
    for env_name in env_names:
        env = gym.make(env_name, **argparse.prepare_env_kwargs(args, gazebo_multienv=True))
        args.port_gazebo = str(int(args.port_gazebo) + 1)
        args.port_ros = str(int(args.port_ros) + 1)
        envs.append(env)
    state = [env.reset() for env in envs][0]
    agent = agents.get_agent(args.agent, **prepare_agent_kwargs(args, state, logger, envs[0].action_space.n))
    if args.pretrained:
        print('load pretrained weights: ', args.pretrained)
        agent.load_weights(args.pretrained)
    agent.train()
    saver = NetSaver(args, session_id)

    trainer = create_reinforce_engine(agent, envs, args)

    # trainer.attach(ProgressBar(persist=False)) # Key error 'percentage' after a few k of epochs !?
    trainer.attach(saver)
    trainer.attach(logger)

    engine_state = trainer.run(data=MultienvStepGenerator(envs, agent, max_steps=args.max_steps),
                               max_epochs=args.epochs_count)


if __name__ == '__main__':
    args = argparse.parse_train_args()
    main(args)
