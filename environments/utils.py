import math
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

def rot_gazebo_transform(orientation):
    orient_tuple = (orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w)
    orientation = euler_from_quaternion(orient_tuple)
    rot = orientation[-1]  # z axis
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
