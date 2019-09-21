import ignite.engine


class NetSaver(object):
    def __init__(self, args, session_id):
        self.sess_dir = session_id
        self.save_period = args.save_period
        # TODO args in yaml
        with open('out/{}/args.txt'.format(self.sess_dir), 'w') as args_file:
            for arg in vars(args):
                args_file.write('{}:{}\n'.format(str(arg), str(getattr(args, arg))))

    def save_net(self, agent, i_episode, reward):
        agent.save_qnet('out/{}/model_{}_total_reward_{}'.format(self.sess_dir,
                                                                 i_episode,
                                                                 reward))

    def attach(self, engine):
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.on_epoch_end)
        engine.add_event_handler(ignite.engine.Events.COMPLETED,
                                 lambda engine: engine.agent.save_qnet('model_final'))

    def on_epoch_end(self, engine):
        if engine.state.epoch != 0 and (engine.state.epoch + 1) % self.save_period == 0:
            self.save_net(engine.state.agent, engine.state.epoch + 1,
                          engine.state.total_reward)

        if engine.state.total_reward > engine.state.max_reward:
            engine.state.max_reward = engine.state.total_reward
            self.save_net(engine.state.agent, engine.state.epoch + 1,
                          engine.state.total_reward)
