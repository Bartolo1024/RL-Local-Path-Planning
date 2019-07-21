import PIL.Image
import math
import numpy as np
import cv2
import environments.utils as utils
import reward.tools.crate_point_list_path as cp
import utils
import environments.path_planners as path_planners
import environments.utils as envutils

class PathPointsController(object):
    def __init__(self,
                 config,
                 path_planner,
                 tolerance=1,
                 sample_path=False,
                 **kwargs):
        self.env_map = env_map = cv2.cvtColor(cv2.imread(config['env_map']), cv2.COLOR_BGR2RGB)
        self.max_y_idx = config['max_y_idx']
        self.max_x_idx = config['max_x_idx']
        self.scale = config['scale']
        self.x_trans, self.y_trans = config['translation']
        self.tolerance = tolerance
        path_planner.subscribe(self.update_list_path)

    def __call__(self, done, action, state, coordinates):
        pass

    def reset(self):
        pass

    def update_list_path(self, path):
        # print('update path {}'.format(path))
        self.list_path = path
        self.path = np.zeros((self.env_map.shape[0], self.env_map.shape[1]), dtype=np.bool)
        for x, y in self.list_path:
            self.path[y, x] = True

    def _get_actual_coordinates(self, coordinates):
        '''
        x and y are swaped in simulator coordinates
        furthermore x axis is inverted
        '''
        x = coordinates.pose.position.x
        y = coordinates.pose.position.y
        z = coordinates.pose.position.z
        x = int(x * self.scale) + self.x_trans
        y = int(y * self.scale) - self.y_trans + self.max_y_idx
        x = utils.clip(x, self.max_y_idx)
        y = utils.clip(y, self.max_x_idx)
        return y, x

    def _get_gazeboo_coordinates(self, map_x, map_y):
        return envutils.get_gazeboo_coordinates(map_x, map_y, self.scale, self.x_trans, self.y_trans, self.max_y_idx)

    def _get_closest_path_point(self, x, y, metric='city'):
        if metric == 'city':
            return min(self.list_path, key=lambda el: abs(el[0] - x) + abs(el[1] - y))
        elif metric == 'euclidean':
            return min(self.list_path, key=lambda el: math.sqrt((el[0] - x) ** 2 + (el[1] - y) ** 2))
        else:
            raise NotImplementedError

    def _get_closest_path_points(self, x, y):
        index_max = min(xrange(len(self.list_path)), key=lambda idx: abs(self.list_path[idx][0] - x) +
                                                                abs(self.list_path[idx][1] - y))
        index_max_2 = min(xrange(len(self.list_path)), key=lambda idx: abs(self.list_path[idx][0] - x) +
                                                                  abs(self.list_path[idx][1] - y) if idx != index_max else 1000)
        if index_max > index_max_2:
            return self.list_path[index_max_2], self.list_path[index_max]
        else:
            return self.list_path[index_max], self.list_path[index_max_2]
