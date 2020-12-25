import os
import sys
import json 

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from flask import Flask, request, Response, jsonify

from networks.retrieval_networks import RetrievalNet
from utils import get_config

app = Flask(__name__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces_retrieval_gmm.yaml', help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='', help='gpu list')
opts = parser.parse_args()

configs = get_config(opts.config)
device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')

# Setup model and initialize
ret_model = RetrievalNet(configs['ret']).to(device)
state_dict = torch.load(config['ret']['retrieval_checkpoint'], map_location=lambda storage, loc: storage)
ret_model.load_state_dict(state_dict['a'])
ret_model.eval()
print("Model loaded.")

# Load embeddings
image_size = configs['image_size']
transform = [T.CenterCrop(configs['crop_size']),
             T.Resize(image_size),
             T.ToTensor(),
             T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
transform = T.Compose(transform)

# Load train images embeddings and all train images to RAM
img_em_data = json.load(open(configs['ret']['img_emb_path']))
print("Number of image embeddings loaded: {}".format(len(img_em_data)))
img_em = torch.zeros(len(img_em_data), configs['ret']['embed_dim']).float().to(device)
img = torch.zeros(len(img_em_data), 3, image_size, image_size).float()
print("Loading training images and their embeddings")
for idx,(k,v) in enumerate(img_em_data.items()):
    img_em[idx,:] = torch.from_numpy(np.array(v['embedding']))
    image = transform(Image.open(os.path.join(configs['data_root'], k)).convert("RGB"))
    img[idx,:,:,:] = image
    if idx % 10000 == 0:
        print("Loading images and embeddings: {} / {}".format(idx,len(img_em_data))) 
del img_em_data

dist = nn.PairwiseDistance(p=configs['ret']['distance_norm_degree'])


@app.route("/pred", methods=["POST"])
def test():
    cont_feat = request.files["cont"].to(device)
    attr_feat = request.files["attr"].to(device)

    query_em  = ret_model(cont_feat, attr_feat)
    distances = dist(img_em, query_em)
    distances = distances.sort(descending=False)
    results_indices = distances[1][0:num_results]
    results_distances = distances[0][0:num_results]
    retrieved_images = img[results_indices,:,:,:]

    responses = {"retrieved_imgs": retrieved_images}
    return jsonify(responses)

app.run(host="0.0.0.0", port=8008)


