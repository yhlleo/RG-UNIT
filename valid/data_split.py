import os
import numpy as np
import random
from collections import OrderedDict

selected_attributes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
list_attr_path = '../datasets/celeba/list_attr_celeba.txt'

lines = [line.rstrip() for line in open(list_attr_path, 'r')]
all_attr_names = lines[1].split()
attr2idx, idx2attr = {}, {}
attr2images = {}
image2attr = OrderedDict()
    
for i, attr_name in enumerate(all_attr_names):
    attr2idx[attr_name] = i
    idx2attr[i] = attr_name

lines = lines[2:]
for line in lines:
    split = line.split()
    filename = split[0]
    values = split[1:]

    label = []
    for attr_name in selected_attributes:
        idx = attr2idx[attr_name]
        label.append(int(values[idx] == '1'))

    key = ''.join([str(v) for v in label])
    if key not in attr2images:
        attr2images[key] = []
    attr2images[key].append(filename)
    image2attr[filename] = key

print("Num of images:", len(image2attr))

# create test subset
test_set = []
for idx, key in enumerate(image2attr):
    if idx > 2000: break
    if len(attr2images[image2attr[key]]) > 10:
        test_set.append(key)

# save 2000
with open('./list_attr_celeba-val.txt', 'w') as fout:
    for i in range(2000):
        fout.write('{}\t{}\n'.format(test_set[i], image2attr[test_set[i]]))

with open('./list_attr_celeba-all.txt', 'w') as fout:
    for idx, key in enumerate(image2attr):
        if idx < 2000: continue
        fout.write('{}\t{}\n'.format(key, image2attr[key]))

# remove items added into test set
for tt in test_set:
    key = image2attr[tt]
    if tt in attr2images[key]:
        attr2images[key] = list(set(attr2images[key]) - set([tt]))

# sample train subset
percent = 0.25
selected_imgs = []
attr2nums = {}
for key in attr2images:
    attr2nums[key] = len(attr2images[key])

for key in attr2images:
    if attr2nums[key] < 10:
        selected_imgs += attr2images[key]
    else:
        selected_imgs += random.sample(attr2images[key], int(attr2nums[key]*percent))

with open('./list_attr_celeba-0.25.txt', 'w') as fout:
    for im in selected_imgs:
        fout.write(im+'\t'+image2attr[im]+'\n')

# remove items added to selected subset
for tt in selected_imgs:
    key = image2attr[tt]
    if tt in attr2images[key]:
        attr2images[key] = list(set(attr2images[key]) - set([tt]))

for key in attr2images:
    selected_imgs += random.sample(attr2images[key], int(attr2nums[key]*percent))

with open('./list_attr_celeba-0.50.txt', 'w') as fout:
    for im in selected_imgs:
        fout.write(im+'\t'+image2attr[im]+'\n')

# remove items added to selected subset
for tt in selected_imgs:
    key = image2attr[tt]
    if tt in attr2images[key]:
        attr2images[key] = list(set(attr2images[key]) - set([tt]))

for key in attr2images:
    selected_imgs += random.sample(attr2images[key], int(attr2nums[key]*percent))

with open('./list_attr_celeba-0.75.txt', 'w') as fout:
    for im in selected_imgs:
        fout.write(im+'\t'+image2attr[im]+'\n')   

