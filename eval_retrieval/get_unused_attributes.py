# For each image, save the dataset attributes that are not used in our setup
# Will use them to evaluate an image similarity that is not related with the learnt attributes

import json
import random

output_file = '../results/celeba_faces_retrieval/unused_attributes.json'
selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
attr_path = '../../datasets/celeba/list_attr_celeba.txt'

attr2idx = {}
idx2attr = {}
unused_attributes = []
output_data = {}

lines = [line.rstrip() for line in open(attr_path, 'r')]
all_attr_names = lines[1].split()
for att_name in all_attr_names:
    if att_name not in selected_attrs:
        unused_attributes.append(att_name)
print("Num unused attributes: " + str(len(unused_attributes)))

for i, attr_name in enumerate(all_attr_names):
    attr2idx[attr_name] = i
    idx2attr[i] = attr_name

lines = lines[2:]
random.seed(1234)
random.shuffle(lines)
for i, line in enumerate(lines):
    split = line.split()
    filename = split[0]
    values = split[1:]

    label = []
    for attr_name in unused_attributes:
        idx = attr2idx[attr_name]
        label.append(int(values[idx] == '1'))

    output_data[filename] = label

json.dump(output_data, open(output_file,'w'))
print("DONE")