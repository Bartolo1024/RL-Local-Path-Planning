import math
import numpy as np


class DWAAgent():
    def __init__(self,
                max_speed=0.55,
                min_speed=.1,
                max_yawrate=1.0,
                max_accel=1.6,
                max_dyawrate=3.2,
                v_reso=0.15,
                yawrate_reso=0.05,
                dt=0.1,
                predict_time=1.7,
                to_goal_cost_gain=2.4,
                speed_cost_gain=0.1,
                robot_radius=0.18):
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
        self.ob = None # TODO get obstacles
        self.u = np.array([0.0, 0.0])
        self.x = np.array([0.0, 0.0, - math.pi / 2.0, 0.0, 0.0])
        self.trajectory = np.array(self.x) # TODO prev trajectory

    def act(self, state):
        goal = state['target_point_coordinates']
        x_pos, y_pos = state['robot_coordinates']
        rot = state['robot_rotation']
        vel = state['robot_velocity']
        omega = state['robot_omega']
        self.x = np.array([x_pos, y_pos, rot, vel, omega])
        self.traj = np.vstack((self.traj, self.x))  # store state history
        self.u, ltraj = self.dwa_control(self.x, self.u, goal, self.ob)
        # TODO transform u
        print(self.u)
        return self.u

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

    def update(self):
        pass

    @staticmethod
    def motion(x, u, dt):
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_trajectory(self, xinit, v, y):
        x = np.array(xinit)
        traj = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y], self.dt)
            traj = np.vstack((traj, x))
            time += self.dt
        return traj

    def calc_final_input(self, x, u, dw, goal, ob):
        xinit = x[:]
        min_cost = 10000.0
        min_u = u
        min_u[0] = 0.0
        best_traj = np.array([x])

        # evalucate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self.v_reso):
            for y in np.arange(dw[2], dw[3], self.yawrate_reso):
                traj = self.calc_trajectory(xinit, v, y)

                # calc cost
                to_goal_cost = self.calc_to_goal_cost(traj, goal)
                speed_cost = self.speed_cost_gain * \
                             (self.max_speed - traj[-1, 3])
                ob_cost = self.calc_obstacle_cost(traj, ob) * 3.2
                # print(ob_cost)

                final_cost = to_goal_cost + speed_cost + ob_cost

                # print (final_cost)

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    min_u = [v, y]
                    best_traj = traj

        return min_u, best_traj

    def calc_obstacle_cost(self, traj, ob):
        # calc obstacle cost inf: collistion, 0:free

        skip_n = 2
        minr = float("inf")

        for ii in range(0, len(traj[:, 1]), skip_n):
            for i in range(len(ob[:, 0])):
                ox = ob[i, 0]
                oy = ob[i, 1]
                dx = traj[ii, 0] - ox
                dy = traj[ii, 1] - oy

                r = math.sqrt(dx ** 2 + dy ** 2)
                if r <= self.robot_radius:
                    return float("Inf")  # collision

                if minr >= r:
                    minr = r

        return 1.0 / minr  # OK

    def calc_to_goal_cost(self, traj, goal):
        # calc to goal cost. It is 2D norm.
        goal_magnitude = math.sqrt(goal[0] ** 2 + goal[1] ** 2)
        traj_magnitude = math.sqrt(traj[-1, 0] ** 2 + traj[-1, 1] ** 2)
        dot_product = (goal[0] * traj[-1, 0]) + (goal[1] * traj[-1, 1])
        error = dot_product / (goal_magnitude * traj_magnitude)
        error_angle = math.acos(error)
        cost = self.to_goal_cost_gain * error_angle
        return cost

    def dwa_control(self, x, u, goal, ob):
        dw = self.calc_dynamic_window(x)
        u, traj = self.calc_final_input(x, u, dw, goal, ob)
        return u, traj
