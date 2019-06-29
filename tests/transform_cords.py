import numpy as np
import cv2
import environments.data as data
import environments.env_config as env_config
import environments.path_points_controller as pc
import agents.dwa_agent as dwa


class Pose(object):
    def __init__(self, x=0, y=0, z=0):
        self.position = Position(x, y, z)


class Position(object):
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


class Coordinates(object):
    def __init__(self, x, y):
        self.pose = Pose(x, y)


def imshow(img, name, scale):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, *scale)
    cv2.imshow(name, img)
    cv2.waitKey(0)


def run_transform_test(env_name='myenv-v0'):
    map_path = data.get_map_path(env_name)
    agent = dwa.DWAAgent()
    obs = agent.load_obstacles(env_name, (5, 5))
    env_map = cv2.cvtColor(cv2.imread(map_path), cv2.COLOR_BGR2RGB)
    config = env_config.get_config(env_name)
    p_cont = pc.PathPointsController(**config)
    res = np.zeros_like(env_map)
    for ob in obs:
        gazebo_cords = Coordinates(ob[0], ob[1])
        print('gazebo cords x', gazebo_cords.pose.position.x)
        print('gazebo cords y', gazebo_cords.pose.position.y)
        x, y = p_cont._get_actual_coordinates(gazebo_cords)
        print('actual', x, y)
        res[x, y] = 255
    imshow(env_map, 'base', (300, 300))
    imshow(res, 'res', (300, 300))
