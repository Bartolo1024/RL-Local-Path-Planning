import os

class NetSaver(object):
    def __init__(self, agent, args, session_id):
        self.agent = agent
        self.sess_dir = session_id
        with open('out/{}/args.txt'.format(self.sess_dir), 'w') as args_file:
            for arg in vars(args):
                args_file.write('{}:{}\n'.format(str(arg), str(getattr(args, arg))))
            args_file.write('gazebo master uri: {}\n'.format(os.environ['GAZEBO_MASTER_URI']))
            args_file.write('gazebo master uri: {}\n'.format(os.environ['ROS_MASTER_URI']))

    def save_net(self, i_episode, reward):
        self.agent.save_qnet('out/{}/model_{}_total_reward_{}'.format(self.sess_dir,
                                                                      i_episode,
                                                                      reward))
