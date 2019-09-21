import sys
import gym
import pathlib2
import agents
import ignite.engine
import args as argument_parser
from utils import path_plot
from utils.step_generator import StepGenerator
from ignite.contrib.handlers import ProgressBar
from utils.lap_time_measure import LapTimeMeasure
from utils.state_recorder import StateRecorder
from utils.action_recorder import ActionRecorder
from environments import GazeboCircuitTurtlebotLidarEnv # must be imported to register in gym


def create_validation_engine(agent, environment):
    def _run_single_simulation(engine, timestep=None):
        transition = engine.state.batch
        state, action, _, reward = transition
        engine.state.total_reward += sum(reward.values())
        engine.state.state = state
        engine.state.action = action
        engine.state.reward = reward

    engine = ignite.engine.Engine(_run_single_simulation)

    @engine.on(ignite.engine.Events.STARTED)
    def initialize(engine):
        engine.state.agent = agent
        engine.state.environment = environment
        engine.state.max_reward = -sys.maxint

    @engine.on(ignite.engine.Events.EPOCH_STARTED)
    def _reset(engine):
        engine.state.total_reward = 0
        engine.state.loss = 0.

    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def _reset_data(engine):
        engine.state.dataloader.reset()

    @engine.on(ignite.engine.Events.COMPLETED)
    def close(_):
        environment.close()

    def _attach(plugin):
        plugin.attach(engine)

    engine.attach = _attach

    return engine


def main(args):
    env_kwargs = argument_parser.prepare_env_kwargs(args)
    env = gym.make(args.environment_name, **env_kwargs)
    state = env.reset()
    agent = agents.get_agent(args.agent, env_name=args.environment_name,
                             network_architecture=args.value_estimator,
                             init_state=state, num_of_actions=env.action_space.n)
    if args.agent in ['dqn', 'a2c']:
        agent.load_weights(args.weights)
        out_path = pathlib2.Path('/'.join(args.weights.split('/')[:-1]))
        agent.eval()
    else:
        out_path = pathlib2.Path('out/{}'.format(args.agent))
        agent.set_action_space(env_kwargs['action_space'])

    evaluator = create_validation_engine(agent, env)

    path_plotter = path_plot.Plotter(args.environment_name, out_path)
    state_recorder = StateRecorder(args.environment_name, out_path)
    action_recorder = ActionRecorder(args.environment_name, out_path, env_kwargs['action_space'])
    evaluator.attach(state_recorder)
    evaluator.attach(action_recorder)
    evaluator.attach(path_plotter)
    evaluator.attach(LapTimeMeasure(out_path, args.environment_name))
    # evaluator.attach(ProgressBar(persist=False))

    engine_state = evaluator.run(data=StepGenerator(env, agent, max_steps=args.max_steps,
                                                    break_if_collision=args.break_if_collision),
                                 max_epochs=1000)


if __name__ == '__main__':
    args = argument_parser.parse_eval_args()
    main(args)
