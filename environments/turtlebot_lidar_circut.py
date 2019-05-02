from builtins import enumerate

import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from reward import get_reward
import state_transforms.target_point_controller as tp_controller

class GazeboCircuitTurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        gazebo_env.GazeboEnv.__init__(self, 'GazeboCircuitTurtlebotLidar_v0.launch')
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)

        self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.model = GetModelStateRequest()
        self.model.model_name = 'mobile_base'
        # self.rewards = get_reward('HitAccBasedReward,PathReward') # default
        self.min_range = 0.2
        self.target_point_controller = tp_controller.TargetPointController()
        self.logger = None
        self._seed()

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges) / new_ranges
        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if data.ranges[i] == float ('Inf'):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges, done

    def transform_observation(self, data, coordinates=None):
        ranges = []
        done = False
        for item in data.ranges:
            if item == float('Inf'):
                ranges.append(1.0)
            elif np.isnan(item):
                ranges.append(0.0)
            else:
                ranges.append(float(item) / 6)
            if (self.min_range > item > 0):
                done = True
        ranges = np.array(ranges, dtype=np.float32).reshape(1, len(ranges))
        if coordinates and self.target_point_controller:
            target_point = self.target_point_controller(coordinates)
            ret = {'ranges': ranges, 'target_point': target_point}
        else:
            ret = {'ranges': ranges, 'target_point': (0., 0.)}
        return ret, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.wait_for_service('/gazebo/unpause_physics', self.unpause)

        if action == 0: #FORWARD
            self.publish_velocity(0.3, .0)
        elif action == 1: #LEFT
            self.publish_velocity(0.05, 0.3)
        elif action == 2: #RIGHT
            self.publish_velocity(0.05, -0.3)

        data = self.wait_for_data('/scan', LaserScan, 5)
        coordinates = self.model_coordinates(self.model)
        self.wait_for_service('/gazebo/pause_physics', self.pause)
        state, done = self.transform_observation(data, coordinates)
        reward = self.get_reward(done, action, state, coordinates)
        return state, reward, done, {}

    def publish_velocity(self, x, z):
        vel_cmd = Twist()
        vel_cmd.linear.x = x
        vel_cmd.angular.z = z
        self.vel_pub.publish(vel_cmd)

    def get_reward(self, done, action, state, coordinates=None):
        reward_values = [rew(done, action, state, coordinates) for rew in self.rewards]
        if self.logger:
            for rew, val in zip(self.rewards, reward_values):
                self.logger.log_value(rew.__class__.__name__, val)
        return {rew.__class__.__name__: val for rew, val in zip(self.rewards, reward_values)}

    def reset(self):
        self.wait_for_service('/gazebo/reset_simulation', self.reset_proxy) # Unpause simulation to make observation
        self.wait_for_service('/gazebo/unpause_physics', self.unpause) #read laser data
        data = self.wait_for_data('/scan', LaserScan, 5)
        self.wait_for_service('/gazebo/pause_physics', self.pause)
        state, done = self.transform_observation(data)
        for rew in self.rewards:
            rew.reset()
        return state, done

    @staticmethod
    def wait_for_service(service_name, try_function):
        rospy.wait_for_service(service_name)
        try:
            try_function()
        except (rospy.ServiceException) as e:
            print(e)
            print('{} service call failed'.format(service_name))

    @staticmethod
    def wait_for_data(service_name, service_lib, time_out):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(service_name, service_lib, timeout=time_out)
            except:
                pass
        return data

    def set_rewards(self, rew_str):
        '''
        :param rew_str: rewards classnames
        ex. HitReward,PathReward,CoinReward
        '''
        self.rewards = get_reward(rew_str)

    def set_logger(self, logger):
        self.logger = logger
