import math
import environments.utils as utils
import environments.path_points_controller as pc


class TargetPointController(pc.PathPointsController):
    def __init__(self,
                 config,
                 path_planner,
                 **kwargs):
        super(TargetPointController, self).__init__(path_planner=path_planner, config=config,
                                                    tolerance=config['tolerance'], sample_path=True, **kwargs)
        self.path_planner = path_planner
        self.path_planner()
        self.point_generator = self._target_point_generator()
        self.next_point = self.point_generator.next()
        self.done = False

    def __call__(self, coordinates, **kwargs):
        x_robot_position, y_robot_position = self._get_actual_coordinates(coordinates)
        # print('points ', (x_robot_position, y_robot_position), self.next_point)
        while utils.distance_between_points((x_robot_position, y_robot_position), self.next_point) < self.tolerance:
            self.next_point = self.point_generator.next()
        target_x, target_y = self.next_point
        tp_direction = self.get_target_point_direction(target_x, target_y, x_robot_position, y_robot_position)
        tp_activation_map = self.get_target_point_activation_map(target_x, target_y, coordinates.pose.orientation)
        tp_gazebo_coordinates = self._get_gazeboo_coordinates(target_x, target_y)
        return {'target_point': tp_direction,
                'target_point_coordinates': tp_gazebo_coordinates,
                'target_point_map_coordinates': (target_x, target_y),
                'robot_position': (x_robot_position, y_robot_position),
                'tp_activation_map': tp_activation_map}

    @staticmethod
    def get_target_point_direction(target_x, target_y, x_robot_position, y_robot_position):
        vec_norm = math.sqrt((target_x - x_robot_position) ** 2 + (target_y - y_robot_position) ** 2)
        ret = (target_x - x_robot_position) / vec_norm, (target_y - y_robot_position) / vec_norm
        return ret

    def get_target_point_activation_map(self, target_x, target_y, orientation, resolution=10):
        import numpy as np
        sim_target_x, sim_target_y = utils.get_gazeboo_coordinates(target_x, target_y, self.scale, self.x_trans,
                                                                   self.y_trans, self.max_y_idx)
        angle = utils.angle_with_x_axis((sim_target_x, sim_target_y))
        robot_orientation = utils.rot_gazebo_transform(orientation)
        diff = int((angle - robot_orientation + 1) * resolution)
        clamp = lambda idx: max(0, min(idx, resolution - 1))
        ret = np.zeros((2 * resolution,))
        ret[clamp(diff - 1)] = .5
        ret[clamp(diff + 1)] = .5
        ret[clamp(diff)] = 1.
        return ret

    def _target_point_generator(self):
        while True:
            for point in self.list_path:
                yield point
            # print('new round')
            self.done = True
            self.path_planner(begin=(point[1], point[0]), random_end=True)

    def reset(self):
        super(TargetPointController, self).reset()
        self.path_planner()
        self.point_generator = self._target_point_generator()
        self.next_point = self.point_generator.next()
        self.done = False
