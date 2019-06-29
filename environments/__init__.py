import logging
from gym.envs.registration import register
from .turtlebot_lidar_circut import GazeboCircuitTurtlebotLidarEnv
from .turtlebot_lidar_maze import GazeboMazeTurtlebotLidarEnv

logger = logging.getLogger(__name__)

register(
    id='myenv-v0',
    entry_point='environments:GazeboCircuitTurtlebotLidarEnv',
)

register(
    id='env-maze-v0',
    entry_point='environments:GazeboMazeTurtlebotLidarEnv',
)
