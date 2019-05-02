from .turtlebot_lidar_circut import GazeboCircuitTurtlebotLidarEnv
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='myenv-v0',
    entry_point='environments:GazeboCircuitTurtlebotLidarEnv',
)

