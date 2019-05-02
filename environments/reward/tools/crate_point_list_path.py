import argparse
import json
import numpy as np
import cv2

def prepare_argparser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--map', type=str, default='environments/data/path.png')
    parser.add_argument('--out', type=str, default='environments/data/path.json')
    parser.add_argument('--start_point', type=str, default=(26, 4))
    return parser

def parse_args():
    parser = prepare_argparser()
    ret = parser.parse_args()
    return ret

def get_colour_map(img, r, g, b):
    ch_r = img[:, :, 0] == r
    ch_g = img[:, :, 1] == g
    ch_b = img[:, :, 2] == b
    return np.logical_and(ch_r, ch_g, ch_b)

def get_path(img, start_point):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1
    path = contours[0]
    path = [(int(el[0][0]), int(el[0][1])) for el in path]
    start = path.index(start_point)
    return path[start:] + path[:start]

def main(args):
    img = cv2.imread(args.map)
    path = get_path(img, args.start_point)
    with open(args.out, 'w') as file:
        json.dump(path, file)

def run_default():
    parser = prepare_argparser()
    args = parser.get_default()
    main(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
