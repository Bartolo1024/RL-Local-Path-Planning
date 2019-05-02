import actor_critic_agent
import dqn_agent

def get_agent(name, *args, **kwargs):
    if name is 'dqn':
        return dqn_agent.DQNAgent(*args, **kwargs)
    elif name is 'a2c':
        return  actor_critic_agent.A2CAgent(*args, **kwargs)
    else:
        raise NotImplementedError('agent {} not implemented'.format(name))
