# --------------------------------------------------------
# RG-UNIT
# Written by Yahui Liu (yahui.liu@unitn.it)
# --------------------------------------------------------

import random
random.seed(1234)

def load_celeba_files(file_path, mode):
    lines = [line.rstrip() for line in open(file_path, 'r')]
    dataset = []
    for ll in lines:
        fname, label = ll.split()
        label = [int(v) for v in label]
        dataset.append([fname, label])
    print('Finished loading {} {} images...'.format(len(dataset), mode))
    random.shuffle(dataset)
    return dataset

def load_cat2dog_files(file_path, mode):
    lines = [line.rstrip() for line in open(file_path, 'r')]
    dataset = []
    for ll in lines:
        fname, label = ll.split()
        dataset.append([fname, int(label)])
    print('Finished loading {} {} images...'.format(len(dataset), mode))
    random.shuffle(dataset)
    return dataset