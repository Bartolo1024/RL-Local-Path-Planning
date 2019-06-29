import random
import numpy as np
import cv2
import a_star
from environments import utils


class RandomPathPlanner(object):
    def __init__(self, env_config):
        self.env_map = cv2.cvtColor(cv2.imread(env_config['env_map']), cv2.COLOR_BGR2RGB)
        begin = np.nonzero(utils.get_colour_map(self.env_map, 255, 0, 0))
        self.begin = begin[0][0], begin[1][0]
        self.env_map[begin[0], begin[1], :] = (255, 255, 255)
        kernel_size = env_config['kernel_size']
        self.env_map_er = cv2.erode(self.env_map,
                                    np.ones((kernel_size, kernel_size), np.uint8),
                                    iterations=1)
        self.obstacles_map = utils.get_colour_map(self.env_map_er, 0, 0, 0)
        self.obstacles_list = [(r, c) for r, c in zip(*np.nonzero(self.obstacles_map))]
        # cv2.imwrite('all_map.png', self.env_map)
        self.allowed_cells_map = cv2.erode(self.env_map_er,
                                           np.ones((3, 3), np.uint8),
                                           iterations=1)
        # cv2.imwrite('allowed_cells.png', self.allowed_cells_map)
        self.allowed_cells_map = utils.get_colour_map(self.allowed_cells_map, 255, 255, 255)
        self.allowed_cells_list = [(r, c) for r, c in zip(*np.nonzero(self.allowed_cells_map))]
        self.solver = a_star.AStar(env_config['max_y_idx'], env_config['max_x_idx'])
        self.subscribents = []

    def __call__(self, begin=None, sample=False):
        if not begin:
            begin = self.begin
        ret = []
        while len(ret) < 10:
            self.solver.init_grid(self.obstacles_list)
            end = self.allowed_cells_list[random.randint(0, len(self.allowed_cells_list))]
            self.solver.set_task(begin, end)
            ret = self.solver.solve()
            self.solver.reset()
        # map = np.zeros_like(self.env_map_er)
        # for obs_x, obs_y in self.obstacles_list:
        #     map[obs_x, obs_y, 0] = 255
        # for obs_x, obs_y in self.allowed_cells_list:
        #     map[obs_x, obs_y, 2] = 255
        # map[end[0], end[1], :] = (0, 255, 255)
        # map[begin[0], begin[1], :] = (0, 255, 0)
        # self.idx += 1
        # cv2.imwrite('path_{}.png'.format(self.idx), map)
        # for y, x in ret[1:-1]:
        #     map[y, x, :] = (255, 255, 255)
        # cv2.imwrite('path_{}.png'.format(self.idx), map)
        if sample:
            ret = self.get_jump_points(ret)
        for sub_func in self.subscribents:
            sub_func(ret)
        # return [(x, y) for y, x in ret]

    @staticmethod
    def get_jump_points(path):
        jump_points = []
        jump_points.append(path[0])
        prev_point = path[0]
        for point in path[1:]:
            if point[0] != prev_point[0] and point[1] != prev_point[1] or point == path[-1]:
                jump_points.append(point)
                prev_point = point
        return jump_points

    def subscribe(self, sub_func):
        self.subscribents.append(sub_func)
