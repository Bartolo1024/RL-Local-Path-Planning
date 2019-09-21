import math
import numpy as np
from tf.transformations import euler_from_quaternion


def clip(idx, max_idx):
    ret = max(0, min(idx, max_idx))
    return ret


def angle_with_x_axis(vector):
    x, y = vector
    y = -y
    if x == 0:
        return 0.5 if y > 0 else -0.5
    angle = math.atan2(float(y), float(x))
    return angle / math.pi


def to_euler_rot_angele(orientation):
    orient_tuple = (orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w)
    orientation = euler_from_quaternion(orient_tuple)
    rot = orientation[-1] / math.pi  # z axis
    return rot


def rot_gazebo_transform(orientation):
    rot = to_euler_rot_angele(orientation)
    rot = rot - 0.5 if rot - 0.5 > -1. else 1. + (rot + 0.5)
    return rot


def compute_distance(point1, point2, x, y):
    'compute using Heron claim'
    if point1 == [x, y] or point2 == [x, y]:
        return 0
    a = math.sqrt(sum(map(lambda el: el * el, points_to_vec(point1, point2))))
    b = math.sqrt(sum(map(lambda el: el * el, points_to_vec(point1, (x, y)))))
    c = math.sqrt(sum(map(lambda el: el * el, points_to_vec(point2, (x, y)))))
    p = .5 * (a + b + c)
    s = math.sqrt(p * (p - a) * (p - b) * (p - c))
    return 2 * s / a

def points_to_vec(p1, p2):
    assert len(p1) == 2
    assert len(p2) == 2
    return (p1[i] - p2[i] for i in range(len(p1)))


def distance_between_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_colour_map(img, r, g, b):
    ch_r = img[:, :, 0] == r
    ch_g = img[:, :, 1] == g
    ch_b = img[:, :, 2] == b
    return np.logical_and(np.logical_and(ch_r, ch_g), ch_b)


def get_map_coordinates(gazebo_coordinates,
                        scale,
                        x_trans, y_trans,
                        max_x_idx, max_y_idx):
    '''
    x and y are swaped in simulator coordinates
    furthermore x axis is inverted
    '''
    x = gazebo_coordinates.pose.position.x
    y = gazebo_coordinates.pose.position.y
    z = gazebo_coordinates.pose.position.z
    x = int(x * scale) + x_trans
    y = int(y * scale) - y_trans + max_y_idx
    x = clip(x, max_y_idx)
    y = clip(y, max_x_idx)
    return y, x


def get_img_coordinates(map_x, map_y,
                        scale,
                        x_trans, y_trans,
                        max_x_idx, max_y_idx):
    '''
    x and y are swaped in simulator coordinates
    furthermore x axis is inverted
    '''
    x = int(map_x * scale) + x_trans
    y = int(map_y * scale) - y_trans + max_y_idx
    x = clip(x, max_y_idx)
    y = clip(y, max_x_idx)
    return y, x


def get_gazeboo_coordinates(map_x, map_y,
                            scale,
                            x_trans, y_trans,
                            max_y_idx):
    x, y = map_y, map_x
    x = (float(x) - x_trans) / scale
    y = (float(y) + y_trans - max_y_idx) / scale
    return x, y


def obstacles_transform(config):
    x_trans, y_trans = config['translation']
    return lambda x, y: get_gazeboo_coordinates(x, y, config['scale'],
                                                x_trans, y_trans,
                                                config['max_y_idx'])


def rev_transform(config):
    x_trans, y_trans = config['translation']
    return lambda x, y: get_img_coordinates(x, y, config['scale'],
                                            x_trans, y_trans,
                                            config['max_x_idx'],
                                            config['max_y_idx'])
