import cv2


def get_config(env_name):
    if env_name == 'myenv-v0':
        env_map='environments/data/env_map.png'
        img = cv2.cvtColor(cv2.imread(env_map), cv2.COLOR_BGR2RGB)
        max_y_idx = img.shape[0] - 1
        max_x_idx = img.shape[1] - 1
        config = dict(env_map=env_map,
                      translation=(4, 2),
                      scale=5,
                      kernel_size=3,
                      max_x_idx=max_x_idx,
                      max_y_idx=max_y_idx)
        return config
    if env_name == 'env-maze-v0':
        env_map='environments/data/map_maze_min_tar.png'
        img = cv2.cvtColor(cv2.imread(env_map), cv2.COLOR_BGR2RGB)
        max_y_idx = img.shape[0] - 1
        max_x_idx = img.shape[1] - 1
        config = dict(env_map=env_map,
                      translation=(23, 21),
                      scale=12,
                      kernel_size=7,
                      max_x_idx=max_x_idx,
                      max_y_idx=max_y_idx)
        return config
    raise NotImplementedError
