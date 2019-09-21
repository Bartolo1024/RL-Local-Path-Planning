import ignite.engine
import analysis.plot_rewards as plot_rewards

class Logger(object):
    def __init__(self, session_id, plot_results=False):
        self.session_id = session_id
        self.file = open('out/{}/log.txt'.format(self.session_id), 'w')
        self.plot_results = plot_results
        self.episode = 0

    def update(self, engine):
        episode = engine.state.epoch
        reward = engine.state.total_reward
        loss = engine.state.loss
        msg = 'episode:{};reward:{};loss{}\n'.format(episode, reward, loss)
        self.file.write(msg)
        if episode != self.episode:
            print('episode {}, reward {}'.format(episode, reward))
            self.episode = episode
            if self.plot_results and episode % 1000 == 0:
                plot_rewards.run(session_id=self.session_id)

    def compute_metrics(self):
        max_reward = []

    def close(self, engine):
        self.file.close()

    def log_value(self, name, value):
        msg = 'episode:{};value:{}\n'.format(self.episode, value)
        with open('out/{}/{}_value.txt'.format(self.session_id, name), 'a+') as file:
            file.write(msg)

    def attach(self, engine):
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.update)
        engine.add_event_handler(ignite.engine.Events.COMPLETED, self.close)
