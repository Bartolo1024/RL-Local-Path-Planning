import math

class Epsilon(object):
    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=1000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.epoch = 0

    def __call__(self):
        eps_th = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * float(self.epoch) / self.eps_decay)
        return eps_th

    def update(self):
        self.epoch += 1
