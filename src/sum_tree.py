import numpy as np
import math

class SumTree:
    def __init__(self, max_entries):
        self.max_entries = max_entries

        self.tree = np.zeros(2 * max_entries - 1)

        self.get_parent = lambda idx: math.floor((idx - 1) / 2)
        self.get_left = lambda idx: 2 * idx + 1
        self.entries_offset = lambda: self.max_entries - 1
    
    def total(self):
        return self.tree[0]
    
    def update(self, idx, priority):
        idx = idx + self.entries_offset()

        delta = priority - self.tree[idx]
        self.tree[idx] = priority

        self.propagate(idx, delta)

    def propagate(self, idx, delta):
        parent = self.get_parent(idx)
        self.tree[parent] += delta
        
        if parent > 0:
            self.propagate(parent, delta)

    def sample(self, x, idx=0):
        left, right = (self.get_left(idx), self.get_left(idx) + 1)

        if left >= len(self.tree):
            return idx - self.entries_offset(), self.tree[idx]

        if x <= self.tree[left]:
            return self.sample(x, idx=left)
        else:
            return self.sample(x - self.tree[left], idx=right)