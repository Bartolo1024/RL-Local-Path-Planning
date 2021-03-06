import cv2
import matplotlib.pyplot as plt
import ignite.engine
import environments.env_config as config
from utils import plugin


class Plotter(plugin.BasePlugin):
    def __init__(self, env_name, out_path):
        super(Plotter, self).__init__(env_name, out_path)
        self.out_path = out_path
        self.config = config.get_config(env_name)
        self.env_map = cv2.cvtColor(cv2.imread(self.config['env_map']), cv2.COLOR_BGR2RGB)
        self.max_reward = 0
        self.reward_sum = 0
        self.reset()

    def update(self, state, reward):
        x_robot_position, y_robot_position = state['robot_position']
        x_target, y_target = state['target_point_map_coordinates']
        self.tmp[y_robot_position, x_robot_position, :] = (0, 255, 0)
        self.tmp[y_target, x_target, :] = (0, 0, 255)
        self.reward_sum += sum(reward.values())

    def reset(self, *_):
        if self.reward_sum > self.max_reward or self.reward_sum > 0:
            out_path = self.out_path / 'best_path_{}.png'.format(self.reward_sum)
            plt.imsave(str(out_path), self.tmp)
            self.max_reward = self.reward_sum
        self.reward_sum = 0
        self.tmp = self.env_map.copy()

    def _update_from_engine(self, engine):
        self.update(engine.state.state, engine.state.reward)

    def attach(self, engine):
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self._update_from_engine)
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.reset)
