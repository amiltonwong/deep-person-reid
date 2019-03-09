from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler

from IPython import embed

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        # create a dictionary with 'list' attribute
        # e.g. defaultdict(list, {0: ['a'], 1: ['b', 'c']})
        self.index_dic = defaultdict(list)
        #       [(img_path, pid, camid)]
        for index, (_, pid, _) in enumerate(data_source): # index: idx of image, pid : person id of image
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        #embed()

    def __iter__(self): # returns an whole epoch data , not batch. e.g. 751 x 4 imgs = 3004, returns a list with a length of 3004
        # batch_size = 32 (N=8, K=4)
        indices = torch.randperm(self.num_identities) # shuffle a list [0..750]
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True # considering not enough imgs (< 4)
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t) # extended by a list
        #embed()
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


if __name__ == '__main__':
    from data_manager import Market1501
    dataset = Market1501(root='/data2')
    sampler = RandomIdentitySampler(dataset.train)
    a = sampler.__iter__()