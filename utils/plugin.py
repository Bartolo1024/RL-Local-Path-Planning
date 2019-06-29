import abc
import environments.env_config as config


class BasePlugin(object):
    def __init__(self, env_name, out_path):
        self.out_path = out_path
        self.config = config.get_config(env_name)

    @abc.abstractmethod
    def update(self, state, reward):
        pass

    @abc.abstractmethod
    def reset(self):
        pass
