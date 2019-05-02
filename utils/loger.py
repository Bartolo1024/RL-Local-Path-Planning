import analysis.plot_rewards as plot_rewards

class Logger(object):
    def __init__(self, session_id, plot_results=True):
        self.session_id = session_id
        self.file = open('out/{}/log.txt'.format(self.session_id), 'w')
        self.plot_results = plot_results
        self.episode = 0

    def update(self, episode, reward):
        msg = 'episode:{};reward:{}\n'.format(episode, reward)
        print(msg)
        self.file.write(msg)
        if episode != self.episode:
            self.episode = episode
            if self.plot_results and episode % 1000 == 0:
                plot_rewards.run(session_id=self.session_id)

    def compute_metrics(self):
        max_reward = []

    def close(self):
        self.file.close()

    def log_value(self, name, value):
        msg = 'episode:{};value:{}\n'.format(self.episode, value)
        with open('out/{}/{}_value.txt'.format(self.session_id, name), 'a+') as file:
            file.write(msg)
