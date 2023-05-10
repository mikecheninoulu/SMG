import os
import numpy as np
from numpy.lib.format import open_memmap

paris = {
    'ntu/xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),(14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'MG-RGB': (
        (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (1, 0), (0, 15), (0, 16),
        (4, 1), (7, 1), (4, 5), (7, 2), (4, 2), (7, 5), (4, 15), (7, 16)),

    'kinetics': ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                 (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
}

sets = {
    'train', 'test'
}

# 'ntu/xview', 'ntu/xsub',  'kinetics'
datasets = {
    'MG-RGB',
}
# bone
from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data = np.load('./data/{}/{}_data.npy'.format(dataset, set))
        N, C, T, V, M = data.shape
        print(N) # gesture index
        print(C) # channel 3
        print(T) # frames
        print(V) # joints
        print(M) # person

        # fp_sp[:, :C, :, :, :] = data
        # for v1, v2 in tqdm(paris[dataset]):
        print(data[50, :, 10, :, 0])
             # print(data[50:60, :, 10, v2, 0])
            # fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
