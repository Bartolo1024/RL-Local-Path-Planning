import re
import numpy as np
from .session import get_new_session_id
from .loger import Logger
from .saver import NetSaver


def _generate_action_space(speed_range=(.05, 0.3), speed_steps=2, rot_range=(-.3, .3), rot_steps=3):
    _speeds = np.linspace(*(speed_range + (speed_steps,)))
    _rots = np.linspace(*(rot_range + (rot_steps,)))
    speeds = np.tile(_speeds, len(_rots))
    rots = np.repeat(_rots, len(_speeds))
    space = tuple((s, r) for s, r in zip(speeds, rots))
    return space


def generate_action_space(gen_str):
    ret = ((.3, .0), (.05, .3), (.05, -.3))
    if not gen_str:
        return ret
    match = re.match(r'GEN\((\d+),(\d+)\)', gen_str)
    if match:
        speed_steps = int(match.group(1))
        rot_steps = int(match.group(2))
        ret = _generate_action_space(speed_steps=speed_steps, rot_steps=rot_steps)
    return ret
