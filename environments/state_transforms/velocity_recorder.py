import math
import environments.utils as utils
from tf.transformations import euler_from_quaternion


class VelocityRecorder(object):
    def __init__(self):
        self.reset()

    def update(self, coordinates):
        time_step = coordinates.header.stamp.secs + coordinates.header.stamp.nsecs / 1000000000.
        x_rob = coordinates.pose.position.x
        y_rob = coordinates.pose.position.y
        d_time = time_step - self.prev_time if self.prev_time else None
        self.prev_time = time_step
        vel = self.update_vel(x_rob, y_rob, d_time)
        rot = utils.to_euler_rot_angele(coordinates.pose.orientation)
        ang_vel = self.update_ang_vel(rot, d_time)
        return vel, ang_vel

    def update_vel(self, x_rob, y_rob, d_time):
        if self.prev_coordinates is None or self.prev_time is None:
            vel = 0
        else:
            x_prev, y_prev = self.prev_coordinates
            vel = math.sqrt((x_prev - x_rob) ** 2 + (y_prev - y_rob) ** 2) / d_time
        self.prev_coordinates = x_rob, y_rob
        return vel

    def update_ang_vel(self, rot, d_time):
        if self.prev_rot is None or self.prev_time is None:
            ang_vel = 0
        else:
            prev_rot = self.prev_rot
            diff = abs(self.prev_rot - rot)
            diff = 2 - diff if diff > 1 else diff
            ang_vel = diff / d_time
        self.prev_rot = rot
        return ang_vel

    def reset(self):
        self.prev_coordinates = None
        self.prev_time = None
        self.prev_rot = None
