import json
import ignite.engine
import numpy as np
from utils import plugin


class StateRecorder(plugin.BasePlugin):
    def __init__(self, env_name, out_path):
        super(StateRecorder, self).__init__(env_name, out_path)
        self.env_name = env_name
        self.out_path = out_path
        self.states = []
        self.reward_sum = 0
        self.max_reward_sum = 0

    def update(self, state, reward):
        self.states.append(self._convert_numpy_arrays(state))
        self.reward_sum += sum(reward.values())

    @staticmethod
    def _convert_numpy_arrays(dictionary):
        dictionary = dict((k, v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in dictionary.items())
        return dictionary

    def reset(self, *_):
        print('reward', self.reward_sum)
        if self.reward_sum > self.max_reward_sum or True:
            path = str(self.out_path / 'states_{}_{}.json'.format(self.env_name, self.reward_sum))
            with open(path, 'w') as file:
                json.dump(self.states, file)
            self.max_reward_sum = self.reward_sum
        self.states = []
        self.reward_sum = 0

    def _update_from_engine(self, engine):
        self.update(engine.state.state, engine.state.reward)

    def attach(self, engine):
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self._update_from_engine)
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.reset)
