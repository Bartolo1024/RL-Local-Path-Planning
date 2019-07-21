import reward
import environments.utils as utils
import environments.path_points_controller as pc

class CheckpointDistanceReward(reward.Reward):
    def __init__(self,
                 decrease_distance_reward=.01,
                 **kwargs):
        super(self, CheckpointDistanceReward).__init__()
        self.decrease_distance_reward = decrease_distance_reward
        self.reset()

    def __call__(self, done, action, state, coordinates):
        self.step += 1
        dist = utils.distance_between_points(state['target_point_map_coordinates'],
                                             state['robot_position'])
        diff = dist - self.prev_distance
        self.prev_distance = dist
        if self.step < 3:
            # first - bad coordinates, second prev computed badly
            return 0
        return self.decrease_distance_reward if diff < 0 else 0

    def reset(self):
        self.prev_distance = 0
        self.step = 0
