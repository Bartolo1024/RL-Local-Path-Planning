import ignite.engine
import sys


class LapTimeMeasure(object):
    def __init__(self, out_path, env_name):
        self.out_path = out_path
        self.env_name = env_name
        self.best_time = sys.maxint

    def update(self, engine):
        # print(engine.state.state['time'])
        if engine.state.environment.transform_observation.target_point_controller.done:
            state = engine.state.state
            time = int(state['time'])
            self.best_time = time if time < self.best_time else self.best_time
            with open(str(self.out_path / 'best_time_{}'.format(self.env_name)), 'w') as f:
                # print(self.best_time)
                f.write(str(self.best_time))

    def attach(self, engine):
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self.update)
