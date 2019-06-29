
def get_map_path(env_name):
    if env_name == 'myenv-v0':
        return 'environments/data/env_map.png'
    if env_name == 'env-maze-v0':
        return 'environments/data/map_maze_min_tar.png'
    else:
        raise NotImplementedError