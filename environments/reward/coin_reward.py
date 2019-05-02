from reward import Reward
import environments.utils as utils
import environments.path_points_controller as pc

class CoinReward(Reward, pc.PathPointsController):
    def __init__(self, **kwargs):
        pc.PathPointsController.__init__(self, **kwargs)
        # TODO sample path or compute path

    def __call__(self, done, action, state, coordinates):
        x, y = self._get_actual_coordinates(coordinates)
        beg_y = utils.clip(y - self.tolerance, self.max_y_idx)
        end_y = utils.clip(y + self.tolerance, self.max_y_idx)
        beg_x = utils.clip(x - self.tolerance, self.max_x_idx)
        end_x = utils.clip(x + self.tolerance, self.max_x_idx)
        rew = self.path[beg_y: end_y + 1, beg_x: end_x + 1].any()
        self.path[beg_y: end_y + 1, beg_x: end_x + 1] = False
        if not self.path.any():
            print('new round')
            self.reset()
        return .5 if rew else -.1

    def reset(self):
        pc.PathPointsController.reset(self)
