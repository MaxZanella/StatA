import numpy as np
from torch.utils.data.sampler import Sampler
from typing import List
from collections import defaultdict
from numpy.random import dirichlet


class LabelCorrelatedSampler(Sampler):
    def __init__(self, data_source, gamma, batch_size, slots=None):
        self.label_dict = defaultdict(list)
        self.classes = set()
        for i, item in enumerate(data_source):
            self.label_dict[item.label].append(i)
            self.classes.add(item.label)
        self.labels = list(self.label_dict.keys())
        self.labels.sort()

        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_class = len(self.classes)
        
        
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100
            
        
        

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []
        label_distribution = dirichlet([self.gamma] * self.num_slots, self.num_class)

        for label in self.labels:
            indices = np.array(self.label_dict[label])
            slot_indices = [[] for _ in range(self.num_slots)]

            partition = label_distribution[self.labels.index(label)]
            print('partition', partition)
            for s, ids in enumerate(np.split(indices, (np.cumsum(partition)[:-1] * len(indices)).astype(int))):
                print(s, ids)
                slot_indices[s].extend(ids)

            for s_ids in slot_indices:
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                for i in permutation:
                    ids.extend(s_ids[i] if isinstance(s_ids[i], list) else [s_ids[i]])
                final_indices.extend(ids)
        print(final_indices)
        return iter(final_indices)

