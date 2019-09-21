class StepGenerator():
    def __init__(self, environment, agent, max_steps, break_if_collision=True):
        self.environment = environment
        self.agent = agent
        self.max_steps = max_steps
        self.break_if_collision = break_if_collision
        self.reset()

    def __iter__(self):
        return self

    def next(self):
        print('done {}'.format(self.done))
        print('tp done {}'.format(self.environment.transform_observation.target_point_controller.done))
        if self.current > self.max_steps or self.environment.transform_observation.target_point_controller.done:
            raise StopIteration
        elif self.done:
            if self.break_if_collision:
                raise StopIteration
            else:
                pass
        action = self.agent.act(self.state)
        prev_state = self.state
        self.state, reward, self.done, _ = self.environment.step(action)
        if self.environment.spec.id == 'CartPole-v0':
            reward = {'Base': reward}
        self.current += 1
        return prev_state, action, self.state, reward

    def __len__(self):
        return self.max_steps

    def reset(self):
        self.state = self.environment.reset()
        self.current = 0
        self.done = False


class MultienvStepGenerator(object):
    def __init__(self, environments, agent, max_steps, break_if_collision=True):
        self.environments = environments
        self.environment = environments[0]
        self.env_idx = 0
        self.agent = agent
        self.max_steps = max_steps
        self.break_if_collision = break_if_collision
        self.reset()

    def __iter__(self):
        return self

    def next(self):
        print(self.environment.transform_observation.target_point_controller.done)
        if self.current > self.max_steps or self.environment.transform_observation.target_point_controller.done:
            print('raise 1')
            raise StopIteration
        elif self.done:
            print('raise done')
            if self.break_if_collision:
                raise StopIteration
            else:
                pass
        action = self.agent.act(self.state)
        prev_state = self.state
        self.state, reward, self.done, _ = self.environment.step(action)
        if self.environment.spec.id == 'CartPole-v0':
            reward = {'Base': reward}
        self.current += 1
        return prev_state, action, self.state, reward

    def __len__(self):
        return self.max_steps

    def reset(self):
        print('reset env', self.environment)
        self.state = self.environment.reset()
        self.env_idx = (self.env_idx + 1) % len(self.environments)
        self.environment = self.environments[self.env_idx]
        print('env changed', self.environment)
        self.current = 0
        self.done = False
