import actor_critic_agent
import dqn_agent
import dwa_agent
import simple_dwa_agent

def get_agent(name, *args, **kwargs):
    if name == 'dqn':
        return dqn_agent.DQNAgent(*args, **kwargs)
    elif name == 'a2c':
        return  actor_critic_agent.A2CAgent(*args, **kwargs)
    elif name == 'dwa':
        return dwa_agent.DWAAgent(*args, **kwargs)
    elif name == 'simple_dwa':
        return simple_dwa_agent.DWAAgent(*args, **kwargs)
    else:
        raise NotImplementedError('agent {} not implemented'.format(name))
