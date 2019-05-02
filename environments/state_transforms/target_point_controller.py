import math
import environments.utils as utils
import environments.path_points_controller as pc

class TargetPointController(pc.PathPointsController):
    def __init__(self, **kwargs):
        super(TargetPointController, self).__init__(**kwargs)
        self.point_generator = self._target_point_generator()
        self.next_point = self.point_generator.next()

    def __call__(self, coordinates, **kwargs):
        x_robot_position, y_robot_position = self._get_actual_coordinates(coordinates)
        closest_point = self._get_closest_path_point(x_robot_position, y_robot_position)
        # cl_idx = self.list_path.index(closest_point)
        # next_idx = cl_idx + 1 if cl_idx + 1 is not len(self.list_path) else 0
        # self.next_point = self.list_path[next_idx]
        if closest_point == self.next_point and \
                utils.distance_between_points(closest_point, self.next_point) < self.tolerance:
            self.next_point = self.point_generator.next()
        target_x, target_y = self.next_point
        vec_norm = math.sqrt((target_x - x_robot_position) ** 2 + (target_y - y_robot_position) ** 2)
        return (target_x - x_robot_position) / vec_norm, (target_y - y_robot_position) / vec_norm

    def _target_point_generator(self):
        while True:
            for point in self.list_path:
                yield point
