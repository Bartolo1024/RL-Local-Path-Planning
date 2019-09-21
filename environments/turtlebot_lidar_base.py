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
from sensor_msgs.msg import LaserScan, Image
from gym.utils import seeding
from reward import get_reward
import connection.utils as con_ut
import state_transforms
import path_planners


class GazeboMazeTurtlebotLidarEnvBase(gazebo_env.GazeboEnv):
    def __init__(self,
                 launch_file,
                 config,
                 port_ros='11311',
                 port_gazebo='11345',
                 reward_str='HitReward,CoinReward',
                 logger=None,
                 randomized_target=True,
                 action_space=((.3, .0), (.05, .3), (.05, -.3))):
        print('port gazeebo: {}, port ros: {}'.format(port_gazebo, port_ros))
        gazebo_env.GazeboEnv.__init__(self,
                                      launch_file,
                                      port=port_ros,
                                      port_gazebo=port_gazebo)
        print('ACTION SPACE', action_space)
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(len(action_space))
        self.action_tuple = action_space
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.model = GetModelStateRequest()
        self.model.model_name = 'mobile_base'
        self.path_planner = path_planners.PathPlanner(config, randomized_target)
        self.rewards = get_reward(reward_str, config, path_planner=self.path_planner)
        self.min_range = 0.2
        self.transform_observation = state_transforms.ObservationTransform(config, self.path_planner)
        self.logger = logger
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        con_ut.wait_for_service('/gazebo/unpause_physics', self.unpause)
        if isinstance(action, tuple) and len(action) == 2:
            self.publish_velocity(*action)
        elif isinstance(action, int):
            self.publish_velocity(*self.action_tuple[action])
        else:
            raise NotImplementedError
        data = con_ut.wait_for_data('/scan', LaserScan, 5)
        # cam = con_ut.wait_for_data('/camera/rgb/image_raw', Image, time_out=5)
        # print(np.array(cam).shape)
        coordinates = self.model_coordinates(self.model)
        con_ut.wait_for_service('/gazebo/pause_physics', self.pause)
        state, done = self.transform_observation(data, coordinates=coordinates)
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
        self.transform_observation.reset()
        con_ut.wait_for_service('/gazebo/reset_simulation', self.reset_proxy) # Unpause simulation to make observation
        con_ut.wait_for_service('/gazebo/unpause_physics', self.unpause) #read laser data
        data = con_ut.wait_for_data('/scan', LaserScan, 5)
        # cam = con_ut.wait_for_data('/camera/rgb/image_raw', Image, time_out=5)
        # print(np.array(cam).shape)
        con_ut.wait_for_service('/gazebo/pause_physics', self.pause)
        state, _ = self.transform_observation(data)
        for rew in self.rewards:
            rew.reset()
        return state
