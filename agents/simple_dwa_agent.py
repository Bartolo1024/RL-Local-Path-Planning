import math
import cv2
import numpy as np
import environments.utils as utils
import environments.data
import environments.utils as envutils
import environments.env_config
import matplotlib.pyplot as plt
from environments import env_config

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


class DWAAgent():
    def __init__(self,
                 env_name='myenv-v0',
                 num_of_actions=3,
                 erode_kernel=(3, 3),
                 max_speed=0.55,
                 min_speed=.1,
                 max_yawrate=1.0,
                 max_accel=1.6,
                 max_dyawrate=3.2,
                 v_reso=0.15,
                 yawrate_reso=0.05,
                 dt=0.5,
                 predict_time=1.7,
                 to_goal_cost_gain=3.4,
                 speed_cost_gain=0.1,
                 robot_radius=0.18,
                 obstacle_cost_gain=1.9,
                 *args,
                 **kwargs):
        self.obstacles = self.load_obstacles(env_name)
        self.reverse_transform = envutils.rev_transform(environments.env_config.get_config(env_name))
        self.num_of_actions = num_of_actions
        self.max_speed = max_speed  # [m/s]
        self.min_speed = min_speed  # [m/s]
        self.max_yawrate = max_yawrate # [rad/s]
        self.max_accel = max_accel  # [m/ss]
        self.max_dyawrate = max_dyawrate  # [rad/ss]
        self.v_reso = v_reso  # [m/s]
        self.yawrate_reso = yawrate_reso  # [rad/s]
        self.dt = dt  # [s]
        self.predict_time = predict_time  # [s]
        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_cost_gain = speed_cost_gain
        self.robot_radius = robot_radius  # [m]
        self.u = np.array([0.0, 0.0])
        self.x = np.array([0.0, 0.0, - math.pi / 2.0, 0.0, 0.0])
        self.trajectory = np.array(self.x)
        self.obstacle_cost_gain = obstacle_cost_gain
        self.prev_time = 0.
        self.action_space = ((.3, .0), (.05, .3), (.05, -.3))

    def load_obstacles(self, env_name):
        config = env_config.get_config(env_name)
        env_map = cv2.cvtColor(cv2.imread(config['env_map']), cv2.COLOR_BGR2RGB)
        self.env_map = env_map = cv2.erode(env_map, np.ones(config['kernel_size'], np.uint8), iterations=1)
        obstacles = utils.get_colour_map(env_map, 0, 0, 0)
        obstacles = [(r, c) for r, c in zip(*np.nonzero(obstacles))]
        config = environments.env_config.get_config(env_name)
        obs_transform = envutils.obstacles_transform(config)
        return np.array([obs_transform(c, r) for r, c in obstacles])

    def act(self, state):
        goal = state['target_point_coordinates']
        x_pos, y_pos = state['robot_coordinates']
        rot = state['robot_rotation']
        vel = state['robot_velocity']
        omega = state['angular_velocity']
        time = state['time']
        if time == 0.:
            self.prev_time = 0.
        self.dt = time - self.prev_time
        print('delta time', self.dt)
        self.prev_time = time
        self.x = np.array([x_pos, y_pos, rot, vel, omega])
        self.u = self.dwa_control(self.x, goal)
        # img = np.zeros_like(self.env_map, dtype=np.uint8)
        # tx, ty = self.reverse_transform(goal[0], goal[1])
        # rx, ry = self.reverse_transform(x_pos, y_pos)
        # img[ty, tx, 0] = 255
        # img[ry, rx, 1] = 255
        # for ob in self.obstacles:
        #     x, y = self.reverse_transform(ob[0], ob[1])
        #     img[y, x, 2] = 255
        # plt.imshow(img)
        # plt.show()
        return tuple(self.u)

    def calc_dynamic_window(self, x):
        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
              -self.max_yawrate, self.max_yawrate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_dyawrate * self.dt,
              x[4] + self.max_dyawrate * self.dt]

        #  [vmin,vmax, yawrate min, yawrate max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def update(self, *args, **kwargs):
        raise RuntimeError('DWA agent is not trainable!')

    @staticmethod
    def motion(x, u, dt):
        next_x = 5 * [0]
        next_x[2] = x[2] + u[1] * dt
        next_x[0] = x[0] + u[0] * math.cos(math.pi * next_x[2]) * dt
        next_x[1] = x[1] + u[0] * math.sin(math.pi * next_x[2]) * dt
        next_x[3] = u[0]
        next_x[4] = u[1]
        return next_x

    def set_action_space(self, action_space):
        self.action_space = action_space

    def get_possible_actions(self, dynamic_window):
        # TODO implement dw grid
        return self.action_space

    def simulate_next_state(self, state, action):
        return self.motion(state, action, self.dt)

    def compute_cost(self, x, goal):
        cost = 0
        mse = lambda p1, p2: math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        distance_from_goal = mse(goal, x)
        cost += max(distance_from_goal, self.robot_radius) * self.to_goal_cost_gain
        nearest_obs = min(self.obstacles, key=lambda obs: mse(obs, x))
        distance_from_obstacles = mse(x, nearest_obs)
        cost += 1 / max(distance_from_obstacles, self.robot_radius) * self.obstacle_cost_gain
        return cost

    def dwa_control(self, x, goal):
        dw = self.calc_dynamic_window(x)
        min_cost = float('Inf')
        min_action = (0., 0.)
        for action in self.get_possible_actions(dw):
            next_x = self.simulate_next_state(x, action)
            x_cost = self.compute_cost(next_x, goal)
            # print('action cost: {}, action {}, next_x: {}'.format(x_cost, action, self.reverse_transform(*next_x[:2])))
            min_cost, min_action = min((x_cost, action), (min_cost, min_action),
                                       key=lambda cost_action: cost_action[0])
        return min_action
