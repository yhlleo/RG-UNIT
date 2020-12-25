# Get closer images for each query
# Save qualitative results 
# Compute P@K, where results contribute to precision with their attributes cosine similarity with the query ones
# Compute P@K using the rest of the attributes, to get some sort of content similarity

import os
import json
import random
import argparse
import operator
import numpy as np
from shutil import copyfile

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_config

random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces_retrieval.yaml', help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu list')
opts = parser.parse_args()
# Load experiment setting
config = get_config(opts.config)

num_results = config['eval']['num_results']  # P@num_results
num_qualitative_results =  config['eval']['num_qualitative']
em_dim = config['ret']['embed_dim']
num_attributes = config['ret']['num_cls']
c_dim = config['c_dim']
device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')

data_root = config['data_root']
ret_model_name = config['eval']['ret_path'].split('/')[-1].strip('.pt')
results_path = os.path.join('results', 
    os.path.splitext(os.path.basename(opts.config))[0],
    ret_model_name)
output_path = results_path + '/retrieval_results/'

queries_em = json.load(open(os.path.join(results_path, 'queries_embeddings.json')))
img_em = json.load(open(os.path.join(results_path, 'images_embeddings.json')))

distance_norm = 2 
dist = nn.PairwiseDistance(p=distance_norm)

# Load unused attributes GT
unused_attributes = json.load(open('datasets/unused_attributes.json'))

# Put img em in a tensor
img_em_tensor = torch.zeros(len(img_em), em_dim, dtype=torch.float32)
img_att = torch.zeros(len(img_em), num_attributes, dtype=torch.float32)
img_extracted_att = torch.zeros(len(img_em), num_attributes, dtype=torch.float32)

def get_mus(style, num_class=8, c_dim=8):
    c_trg = torch.ones(1, num_class).float()
    for i in range(num_class):
        z = style[i*c_dim:(i+1)*c_dim]
        if z.mean() < 0.0:
            c_trg[0, i] = -1.0
    return c_trg

img_names = []
for i,(k,v) in enumerate(img_em.items()):
    img_names.append(k)
    img_em_tensor[i,:] = torch.from_numpy(np.array(v['embedding']))

    cur_img_att = torch.from_numpy(np.array(v['label']))
    cur_img_att[cur_img_att==0] = -1 # Use -1/1 att encoding instead of 0/1 to compute cosine similarity
    img_att[i,:] = cur_img_att

    # Compute also results with extracted attributes (instead of GT)
    cur_extracted_att = torch.from_numpy(np.array(v['extracted_attributes']))
    cur_extracted_att = get_mus(cur_extracted_att, num_attributes, c_dim)
    img_extracted_att[i,:] = cur_extracted_att

img_em_tensor = img_em_tensor.to(device)

# For each query
pure_precision_at = 0
pure_precision_at_content = 0
precision_at = 0
precision_at_content = 0
precision_at_extracted = 0

# Random
pure_precision_at_rnd = 0
pure_precision_at_content_rnd = 0
precision_at_rnd = 0
precision_at_content_rnd = 0
precision_at_extracted_rnd = 0

def cos_sim(a,b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    mult_norm = norma * normb
    if mult_norm == 0:
        mult_norm = 1e-10
    cos = dot / (mult_norm)
    return cos

print("Evaluating ...")
for i, (q_img_name,v) in enumerate(queries_em.items()):
    query_em = torch.from_numpy(np.array(v['embedding'])).float().to(device)
    distances = dist(img_em_tensor, query_em)
    distances = distances.sort(descending=False)
    results_indices = np.array(distances[1][0:num_results].cpu())
    results_scores = np.array(distances[0][0:num_results].cpu())
    results_indices_random = []
    for r in range(num_results):
        results_indices_random.append(random.randint(0,len(img_em)-1))
    results_indices_random = np.array(results_indices_random)

    results_scores_str = ''
    att_cossim = ''
    extracted_att_cossim = ''
    unused_att_cossim = ''

    # Compute P@K
    query_att = np.array(v['trg_label'])
    query_att[query_att==0] = -1
    for r_idx in range(0,num_results):

        results_scores_str += "{:.3f} ".format(results_scores[r_idx])

        # P@K for attributes (using GT)
        cos = cos_sim(query_att, np.array(img_att[results_indices[r_idx]]))
        precision_at += cos
        att_cossim += "{:.3f} ".format(cos)
        # Pure P@K (element is only relevant if it has all the target attributes)
        if np.array_equal(query_att, np.array(img_att[results_indices[r_idx]])):
            pure_precision_at += 1
        # Random 
        cos = cos_sim(query_att, np.array(img_att[results_indices_random[r_idx]]))
        precision_at_rnd += cos
        if np.array_equal(query_att, np.array(img_att[results_indices_random[r_idx]])):
            pure_precision_at_rnd += 1

        # P@K for unused attributes (using GT)
        q_unused = np.array(unused_attributes[q_img_name])
        r_unused = np.array(unused_attributes[img_names[results_indices[r_idx]]])
        q_unused[q_unused==0] = -1
        r_unused[r_unused==0] = -1
        cos = cos_sim(q_unused, r_unused)
        precision_at_content += cos
        unused_att_cossim += "{:.3f} ".format(cos)
        # Pure P@K (element is only relevant if it has all the target attributes)
        if np.array_equal(q_unused, r_unused):
            pure_precision_at_content += 1
        # Random 
        r_unused = np.array(unused_attributes[img_names[results_indices_random[r_idx]]])
        r_unused[r_unused==0] = -1
        cos = cos_sim(q_unused, r_unused)
        precision_at_content_rnd += cos
        # Pure P@K (element is only relevant if it has all the target attributes)
        if np.array_equal(q_unused, r_unused):
            pure_precision_at_content_rnd += 1

        # P@K for attributes (using Ea extracted attributes
        # print(np.array(img_extracted_att[results_indices[r_idx]]))
        cos = cos_sim(query_att, np.array(img_extracted_att[results_indices[r_idx]]))
        precision_at_extracted += cos
        extracted_att_cossim += "{:.3f} ".format(cos)
        # Random 
        cos = cos_sim(query_att, np.array(img_extracted_att[results_indices_random[r_idx]]))
        precision_at_extracted_rnd += cos


    # Save retrieved IMG
    if i < num_qualitative_results:
        out_folder = output_path + q_img_name.strip('.jpg') + '/'
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        # Copy query image
        copyfile(data_root + '/' + q_img_name, out_folder + 'query_img_' + q_img_name)
        # Copy results
        for i,result_idx in enumerate(results_indices):
            copyfile(data_root + '/' + img_names[result_idx], out_folder + str(i) + '_' + img_names[result_idx])

        # Create .txt with text modifier
        txt_file = open(out_folder + 'text_modifier.txt','w')
        #txt_file.write('text_modifier: ' + v['text_modifier'] + '\n')
        txt_file.write('trg_label: ' + str(v['trg_label']) + '\n')
        txt_file.write('results_scores: ' + results_scores_str + '\n')
        txt_file.write('att_cossim: ' + att_cossim + '\n')
        txt_file.write('extracted_att_cossim: ' + extracted_att_cossim + '\n')
        txt_file.write('unused_att_cossim: ' + unused_att_cossim + '\n')

        txt_file.close()

precision_at /= (num_results*len(queries_em))
precision_at_content /= (num_results*len(queries_em))
precision_at_extracted /= (num_results*len(queries_em))
pure_precision_at /= (num_results*len(queries_em))
pure_precision_at_content /= (num_results*len(queries_em))


precision_at_rnd /= (num_results*len(queries_em))
precision_at_content_rnd /= (num_results*len(queries_em))
precision_at_extracted_rnd /= (num_results*len(queries_em))
pure_precision_at_rnd /= (num_results*len(queries_em))
pure_precision_at_content_rnd /= (num_results*len(queries_em))

print("Model: " + str(ret_model_name))
print("Precision at (extracted) " +  str(num_results) + ': ' + str(precision_at_extracted))
print("Precision at " +  str(num_results) + ': ' + str(precision_at))
print("Precision at " +  str(num_results) + ' content : ' + str(precision_at_content))
print("Average Precision at " +  str(num_results) + ' (attributes and content) : ' + str((precision_at + precision_at_content) / 2))
print("---------- Strict Precisions ----------")
print("Strict Precision at " +  str(num_results) + ': ' + str(pure_precision_at))
print("Strict Precision at " +  str(num_results) + ' content : ' + str(pure_precision_at_content))
print("---------- RANDOM -----------")
print("Precision at (extracted) Random " +  str(num_results) + ': ' + str(precision_at_extracted_rnd))
print("Precision at Random " +  str(num_results) + ': ' + str(precision_at_rnd))
print("Precision at Random " +  str(num_results) + ' content : ' + str(precision_at_content_rnd))
print("Average Precision at " +  str(num_results) + ' (attributes and content) : ' + str((precision_at_rnd + precision_at_content_rnd) / 2))
print("Strict Precision at Random " +  str(num_results) + ': ' + str(pure_precision_at_rnd))
print("Strict Precision at Random " +  str(num_results) + ' content : ' + str(pure_precision_at_content_rnd))


print("DONE")

