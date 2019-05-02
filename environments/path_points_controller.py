import PIL
import math
import numpy as np
import cv2
import environments.utils as utils
import reward.tools.crate_point_list_path as cp

class PathPointsController(object):
    def __init__(self,
                 map='environments/data/map.png',
                 path_img='environments/data/path.png',
                 translation=(4, 2),
                 scale=5,
                 tolerance=1,
                 testmode=False):
        img = PIL.Image.open(map)
        img = np.array(img)
        self.img = img
        self.max_y_idx = img.shape[0]- 1
        self.max_x_idx = img.shape[1]- 1
        self.scale = scale
        self.x_trans, self.y_trans = translation
        self.tolerance = tolerance
        self.testmode = testmode
        self.path = self.get_colour_map(self.img, 0, 255, 0)
        self.blocks = self.get_colour_map(self.img, 0, 0, 0)
        img = cv2.imread(path_img)
        self.path_img = path_img
        self.list_path = cp.get_path(img, (26, 4))

    @staticmethod
    def get_colour_map(img, r, g, b):
        ch_r = img[:, :, 0] == r
        ch_g = img[:, :, 1] == g
        ch_b = img[:, :, 2] == b
        return np.logical_and(ch_r, ch_g, ch_b)

    def __call__(self, done, action, state, coordinates):
        pass

    def reset(self):
        self.path = self.get_colour_map(self.img, 0, 255, 0)
        self.blocks = self.get_colour_map(self.img, 0, 0, 0)
        img = cv2.imread(self.path_img)
        self.list_path = cp.get_path(img, (26, 4))

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
