from turtlebot_lidar_base import GazeboMazeTurtlebotLidarEnvBase
import env_config

class GazeboCircuitTurtlebotLidarEnv(GazeboMazeTurtlebotLidarEnvBase):
    def __init__(self,
                 port='11311',
                 port_gazebo='11345',
                 reward_str='HitReward,CoinReward',
                 logger=None):
        config = env_config.get_config('myenv-v0')
        GazeboMazeTurtlebotLidarEnvBase.__init__(self,
                                                 launch_file='GazeboCircuitTurtlebotLidar_v0.launch',
                                                 config=config,
                                                 port=port,
                                                 port_gazebo=port_gazebo,
                                                 reward_str=reward_str,
                                                 logger=logger)
