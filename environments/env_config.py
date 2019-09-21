import cv2


def get_config(env_name):
    if env_name == 'myenv-v0':
        env_map='environments/data/circuit/env_map.png'
        img = cv2.cvtColor(cv2.imread(env_map), cv2.COLOR_BGR2RGB)
        max_y_idx = img.shape[0] - 1
        max_x_idx = img.shape[1] - 1
        config = EnvironmentConfig(env_map=env_map,
                                   translation=(4, 2),
                                   scale=5,
                                   kernel_size=3,
                                   tolerance=5,
                                   max_x_idx=max_x_idx,
                                   max_y_idx=max_y_idx,
                                   endpoints=None)
        return config
    if env_name == 'env-maze-v0':
        env_map='environments/data/maze/cp_map_maze_min_tar.png' # map_maze_min_tar.png'
        img = cv2.cvtColor(cv2.imread(env_map), cv2.COLOR_BGR2RGB)
        max_y_idx = img.shape[0] - 1
        max_x_idx = img.shape[1] - 1
        config = EnvironmentConfig(env_map=env_map,
                                   translation=(23, 21),
                                   scale=10.5,
                                   kernel_size=7,
                                   tolerance=10,
                                   max_x_idx=max_x_idx,
                                   max_y_idx=max_y_idx,
                                   endpoints=None)
        return config
    raise NotImplementedError


class EnvironmentConfig(object):
    def __init__(self, env_map, translation, scale, kernel_size,
                 tolerance, max_x_idx, max_y_idx, endpoints):
        self.env_map = env_map
        self.endpoints = ()
        self.translation = translation
        self.scale = scale
        self.kernel_size = kernel_size
        self.tolerance = tolerance
        self.max_x_idx = max_x_idx
        self.max_y_idx = max_y_idx
        self.endpoints = endpoints

    # TODO replace dict usages and delete this
    def __getitem__(self, item):
        return self.__getattribute__(item)