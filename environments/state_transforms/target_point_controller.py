import math
import environments.utils as utils
import environments.path_points_controller as pc


class TargetPointController(pc.PathPointsController):
    def __init__(self,
                 config,
                 path_planner,
                 tolerance=7,
                 **kwargs):
        super(TargetPointController, self).__init__(path_planner=path_planner, config=config, tolerance=tolerance, sample_path=True, **kwargs)
        self.path_planner = path_planner
        self.path_planner()
        self.point_generator = self._target_point_generator()
        self.next_point = self.point_generator.next()

    def __call__(self, coordinates, **kwargs):
        x_robot_position, y_robot_position = self._get_actual_coordinates(coordinates)
        while utils.distance_between_points((x_robot_position, y_robot_position), self.next_point) < self.tolerance:
            self.next_point = self.point_generator.next()
        target_x, target_y = self.next_point
        tp_direction = self.get_target_point_direction(target_x, target_y, x_robot_position, y_robot_position)
        tp_gazebo_coordinates = self._get_gazeboo_coordinates(target_x, target_y)
        return {'target_point': tp_direction,
                'target_point_coordinates': tp_gazebo_coordinates,
                'target_point_map_coordinates': (target_x, target_y),
                'robot_position': (x_robot_position, y_robot_position)}

    def get_target_point_direction(self, target_x, target_y, x_robot_position, y_robot_position):
        vec_norm = math.sqrt((target_x - x_robot_position) ** 2 + (target_y - y_robot_position) ** 2)
        ret = (target_x - x_robot_position) / vec_norm, (target_y - y_robot_position) / vec_norm
        return ret

    def _target_point_generator(self):
        while True:
            for point in self.list_path:
                yield point

    def reset(self):
        super(TargetPointController, self).reset()
        self.path_planner()
        self.point_generator = self._target_point_generator()
        self.next_point = self.point_generator.next()
