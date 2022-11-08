# from: https://github.com/ZiyuanMa/R2D2/blob/main/priority_tree.py

from typing import Tuple
import numpy as np


class PriorityTree:
    def __init__(self, capacity):
        self.num_layers = 1
        while capacity > 2**(self.num_layers-1): 
            self.num_layers += 1

        self.ptree = np.zeros(2**self.num_layers-1, dtype=np.float64)
    

    def update(self, idxes: np.ndarray, priorities: np.ndarray):

        idxes = idxes + 2**(self.num_layers-1) - 1
        self.ptree[idxes] = priorities

        for _ in range(self.num_layers-1):
            idxes = (idxes-1) // 2
            idxes = np.unique(idxes)
            self.ptree[idxes] = self.ptree[2*idxes+1] + self.ptree[2*idxes+2]


    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        p_sum = self.ptree[0]
        interval = p_sum / num_samples

        prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(0, interval, num_samples)

        idxes = np.zeros(num_samples, dtype=np.int64)
        for _ in range(self.num_layers-1):
            nodes = self.ptree[idxes*2+1]
            idxes = np.where(prefixsums < nodes, idxes*2+1, idxes*2+2)
            prefixsums = np.where(idxes%2 == 0, prefixsums - self.ptree[idxes-1], prefixsums)
        
        # importance sampling weights
        priorities = self.ptree[idxes]

        idxes -= 2**(self.num_layers-1) - 1

        return idxes, priorities
