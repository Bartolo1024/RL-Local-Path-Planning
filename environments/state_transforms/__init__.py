import math
import numpy as np
import target_point_controller as tp_controller
from . import velocity_recorder as vel_rec
import environments.utils as utils

class ObservationTransform(object):
    def __init__(self, config, path_planner, min_range=0.2):
        self.target_point_controller = tp_controller.TargetPointController(config, path_planner)
        self.min_range = min_range
        self.vel_recorder = vel_rec.VelocityRecorder()

    def __call__(self, lidar_data, camera_data=None, coordinates=None):
        ranges = []
        done = False
        for item in lidar_data.ranges:
            if item == float('Inf'):
                ranges.append(1.0)
            elif np.isnan(item):
                ranges.append(0.0)
            else:
                ranges.append(float(item) / 6)
            if (self.min_range > item > 0):
                done = True
        ranges = np.array(ranges, dtype=np.float32).reshape(1, len(ranges))
        ret = {'ranges': ranges}
        ret.update(self.get_robot_kinematic_state(coordinates))
        if coordinates and self.target_point_controller:
            time = coordinates.header.stamp.secs + coordinates.header.stamp.nsecs / 1000000000.
            ret.update(self.target_point_controller(coordinates))
            ret['time'] = time
        else:
            ret.update({'target_point': (0., 0.),
                        'target_point_coordinates': (0., 0.),
                        'target_point_map_coordinates': (0, 0),
                        'robot_position': (0, 0),
                        'time': 0.})
        return ret, done

    def get_robot_kinematic_state(self, coordinates=None):
        ret = {}
        if coordinates:
            ret['robot_coordinates'] = (coordinates.pose.position.x, coordinates.pose.position.y)
            ret['robot_rotation'] = utils.to_euler_rot_angele(coordinates.pose.orientation)
            vel, ang_vel = self.vel_recorder.update(coordinates)
            ret['robot_velocity'] = vel
            ret['angular_velocity'] = ang_vel
        else:
            ret['robot_coordinates'] = (0., 0.)
            ret['robot_rotation'] = -0.5
            ret['robot_velocity'] = 0.
            ret['angular_velocity'] = 0.
        return ret

    def reset(self):
        self.vel_recorder.reset()
        self.target_point_controller.reset()
