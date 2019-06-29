import math
import reward
import environments.utils as utils
import environments.path_points_controller as pc

class PathReward(reward.Reward, pc.PathPointsController):
    def __init__(self,
                 config,
                 path_planner,
                 bias=1.0,
                 **kwargs):
        pc.PathPointsController.__init__(self, config, path_planner, sample_path=False, **kwargs)
        self.prev_coordinates = None
        self.prev_time = None
        self.prev_dist = 0
        self.bias = bias

    def __call__(self, done, action, state, coordinates):
        x_robot_position, y_robot_position = self._get_actual_coordinates(coordinates)
        vel = self._get_velocity(coordinates)
        prev_point, next_point = self._get_closest_path_points(x_robot_position, y_robot_position)
        path_direction_vec = (next_point[i] - prev_point[i] for i in range(2))

        path_angle = utils.angle_with_x_axis(path_direction_vec)
        robot_angle = utils.rot_gazebo_transform(coordinates.pose.orientation)
        ang_diff = abs(path_angle - robot_angle)
        angle_reward = math.cos(ang_diff * math.pi) * vel - math.sin(ang_diff * math.pi) * vel

        dist = utils.compute_distance(prev_point, next_point, x_robot_position, y_robot_position)
        dist_reward = self.prev_dist - dist
        self.prev_dist = dist

        return 10. * angle_reward - .3 * dist

    def _get_velocity(self, coordinates):
        time_step = coordinates.header.stamp.secs + coordinates.header.stamp.nsecs / 1000000000.
        x_rob = coordinates.pose.position.x
        y_rob = coordinates.pose.position.y
        if self.prev_coordinates is None or self.prev_time is None:
            vel = 0
        else:
            x_prev, y_prev = self.prev_coordinates
            d_time = time_step - self.prev_time
            vel = math.sqrt((x_prev - x_rob) ** 2 + (y_prev - y_rob) ** 2) / d_time
        self.prev_coordinates = x_rob, y_rob
        self.prev_time = time_step
        return vel