import json
import ignite.engine


class ActionRecorder(object):
    def __init__(self, env_name, out_path, action_space):
        self.env_name = env_name
        self.out_path = out_path
        self.actions = []
        self.action_space = action_space
        self.reward_sum = 0
        self.max_reward_sum = 0

    def update(self, action, reward):
        self.actions.append(action)
        self.reward_sum += sum(reward.values())

    def reset(self, *_):
        if self.reward_sum > self.max_reward_sum or True:
            path = str(self.out_path / 'actions_{}_{}.json'.format(self.env_name, self.reward_sum))
            with open(path, 'w') as f:
                json.dump(self.actions, f)
            self.max_reward_sum = self.reward_sum
        self.actions = []
        self.reward_sum = 0

    def _update_from_engine(self, engine):
        self.update(engine.state.action, engine.state.reward)

    def attach(self, engine):
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self._update_from_engine)
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.reset)
