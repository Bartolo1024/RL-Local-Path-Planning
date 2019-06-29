import numpy as np
import cv2

class HumanAgent(object):
    def __init__(self, delay=0):
        self.delay = delay
        cv2.imshow('steering window', np.zeros((3, 3, 3), dtype=np.uint8))

    def act(self, state):
        while True:
            key = cv2.waitKey(0)
            if key == ord('w'):
                print('acc')
                acc = 0
            elif key == ord('a'):
                print('left')
                acc = 1
            elif key == ord('d'):
                print('right')
                acc = 2
            else:
                acc = None
            if acc is not None:
                return acc
