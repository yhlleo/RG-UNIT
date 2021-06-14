# --------------------------------------------------------
# RG-UNIT
# Written by Yahui Liu (yahui.liu@unitn.it)
# --------------------------------------------------------

import os
import copy
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

from .losses import triplet_margin
from .models import (
    AdaINGen,
    RetrievalNet
)

from .utils import (
    assign_gmm_componet,
    dist_sampling_split,
    moving_average,
    weights_init,
    print_network,
    debug_image,
    load_checkpoint,
    save_checkpoint,
    load_pretrained_model
)

from data import build_loader, build_dataset, build_ret_loader
from .optims import (
    build_scheduler,
    build_optimizer
)

import random
random.seed(1234)

def g_decoder(decode, x, content, style, use_attention=False):
    x_rec, x_rec_att = decode(content, style)
    if use_attention:
        x_rec = x_rec * x_rec_att + x * (1-x_rec_att)
    return x_rec

def built_triplet(
    r_model, 
    triplets
):
    anc = r_model(triplets[0], triplets[1]) #(anc_cont, anc_sty)
    pos = r_model(triplets[2], triplets[3]) #(pos_cont, pos_sty)
    neg = r_model(triplets[4], triplets[5]) #(neg_cont, neg_sty)
    return anc, pos, neg

def compute_ret_loss(
    r_model, 
    criterion,
    triplets
):  
    anc, pos, neg = built_triplet(r_model, triplets)
    return criterion(anc, pos, neg)

def convert_triplets(inputs, args, g_model, device):
    img_anc, img_neg, lab_src, lab_trg, lab_rnd = inputs
    gmm_src = assign_gmm_componet(
        lab_src, 
        num_cls=args['num_cls'], 
        mode=args['dataset_name']
    ).to(device)
    gmm_trg = assign_gmm_componet(
        lab_trg, 
        num_cls=args['num_cls'], 
        mode=args['dataset_name']
    ).to(device)
    gmm_rnd = assign_gmm_componet(
        lab_rnd, 
        num_cls=args['num_cls'], 
        mode=args['dataset_name']
    ).to(device)

    use_attention = args['gen']['use_attention']

    # collect content and style features from real images, extracted by frozen pretrained model
    img_anc_cont, img_anc_sty = g_model.encode(img_anc)
    img_anc_sty = torch.cat(img_anc_sty, dim=1)
    img_neg_cont, _ = g_model.encode(img_neg)

    # sampling style codes 
    z_trg = dist_sampling_split(gmm_trg, args['attr_dim'], device)
    z_rnd = dist_sampling_split(gmm_rnd, args['attr_dim'], device)

    # translate to target domains
    img_anc2trg = g_decoder(g_model.decode, img_anc, img_anc_cont, z_trg, use_attention)
    # collect content and style features from translated images
    img_anc2trg_cont, img_anc2trg_sty = g_model.encode(img_anc2trg)
    img_anc2trg_sty = torch.cat(img_anc2trg_sty, dim=1)

    triplets = []
    if args['triplets']['easy']:
        triplets.append([
            img_anc_cont, z_trg,
            img_anc2trg_cont, img_anc2trg_sty,  # "translated-reconstructed"
            img_neg_cont, z_rnd                 # both content and style are different
        ])
        
        triplets.append([
            img_anc_cont, z_trg,
            img_anc2trg_cont, img_anc2trg_sty,
            img_neg_cont, z_trg                 # only content is different
        ])
    if args['triplets']['medium']:
        img_neg2trg = g_decoder(g_model.decode, img_neg, img_neg_cont, z_trg, use_attention)
        img_neg2trg_cont, img_neg2trg_sty = g_model.encode(img_neg2trg)
        img_neg2trg_sty = torch.cat(img_neg2trg_sty, dim=1)

        triplets.append([
            img_anc_cont, z_trg,
            img_anc2trg_cont, img_anc2trg_sty,
            img_neg2trg_cont, img_neg2trg_sty
        ])
    if args['triplets']['hard']:
        img_anc2rnd = g_decoder(g_model.decode, img_anc, img_anc_cont, z_rnd, use_attention)
        img_anc2rnd_cont, img_anc2rnd_sty = g_model.encode(img_anc2rnd)
        img_anc2rnd_sty = torch.cat(img_anc2rnd_sty, dim=1)
        triplets.append([
            img_anc_cont, z_trg,
            img_anc2trg_cont, img_anc2trg_sty,
            img_anc2rnd_cont, img_anc2rnd_sty
        ])
    if args['triplets']['hardest']:
        img_anc2rec = g_decoder(g_model.decode, img_anc, img_anc_cont, img_anc_sty, use_attention)
        img_anc2rec_cont, img_anc2rec_sty = g_model.encode(img_anc2rec)
        img_anc2rec_sty = torch.cat(img_anc2rec_sty, dim=1)
        triplets.append([
            img_anc_cont, z_trg,
            img_anc2trg_cont, img_anc2trg_sty,
            img_anc2rec_cont, img_anc2rec_sty
        ])
    return triplets

@torch.no_grad()
def validation(
    val_loader, 
    args,
    r_model,
    g_model,
    margin=0.2,
    norm_degree=2,
    device=None
):
    correct = 0
    all_samples = 0
    for idx, val in enumerate(val_loader):
        val = [v.to(device) for v in val]
        triplets = convert_triplets(val, args, g_model, device)
        for tp in triplets:
            anc, pos, neg = built_triplet(r_model, *tp)
            dis_anc2pos = F.pairwise_distance(anc, pos, p=norm_degree)
            dis_anc2neg = F.pairwise_distance(anc, neg, p=norm_degree)

            correct += torch.sum(dis_anc2neg - dis_anc2pos > margin)
            all_samples += dis_anc2pos.size(0)
    return correct.float()/all_samples

@torch.no_grad()
def collect_embeddings(
    data_loader,
    g_model,
    r_model,
    device=None
): 
    image_embeddings = {}
    for _, data_iter in enumerate(data_loader):
        img, fname, _, lab = data_iter
        img = img.to(device)

        img_cont, img_sty = g_model.encode(img)
        embeddings = r_model(img_cont, img_sty)
        for idx, emb in enumerate(embeddings):
            image_embeddings[fname[idx]] = {}
            image_embeddings[fname[idx]]['emb'] = emb.cpu()
    return image_embeddings


def retrieval_run(args, local_rank):
    print(distributed.get_rank())
    if args['world_size'] > 1:
        torch.manual_seed(1234+distributed.get_rank())
        random.seed(5678+distributed.get_rank())
    device = torch.device('cuda', local_rank)
    mark_flag = args['world_size']==1 or distributed.get_rank() ==0

    # build models
    print("Build and initilize models ...")
    gen_model = AdaINGen(args['gen']).eval().to(device)
    for param in gen_model.parameters():
        param.requires_grad = False

    ret_model = RetrievalNet(args['ret']).to(device)
    print_network(gen_model, "GEN")
    print_network(ret_model, "RET")
    ret_model.apply(weights_init('gaussian'))
    load_pretrained_model(gen_model, args['pretrained_path'])

    # resume training if necessary
    load_checkpoint(ret_model, args['checkpoint_dir'], args['resume_iter'], suffix='ret_ema')
    ret_ema = copy.deepcopy(ret_model)
    
    # define criterion
    criterion = triplet_margin(args['margin'], args['norm_degree'])

    # build dataset and loader
    dataset_train, dataset_test = build_dataset(
        args['image_dir'],
        args['train_list_path'],
        args['test_list_path'],
        args['crop_size'],
        args['img_size'],
        args['dataset_name'],
        args['is_retrieval']
    )
    train_loader, test_loader = build_loader(
        dataset_train,
        dataset_test, 
        args['batch_size'], 
        args['num_workers']
    )

    # build optimizer and schedular
    ret_opt = build_optimizer(ret_model, base_lr=args['lr'])
    n_iter_per_epoch = len(train_loader)
    ret_scheduler = build_scheduler(
        ret_opt, 
        args['total_epochs'], 
        n_iter_per_epoch,
        scheduler_type=args['scheduler_type'],
        warmup_epochs=args['warmup_epochs']
    )
    
    total_steps = args['total_epochs'] * n_iter_per_epoch

    cur_step = 0
    for epoch in range(args['total_epochs']):
        train_loader.sampler.set_epoch(epoch)
        train_data_iter = iter(train_loader)

        start_time = time.time()
        for _ in range(len(train_loader)):
            try:
                # Samples the batch
                inputs = next(train_data_iter)
            except:
                # restart the generator if the previous generator is exhausted.
                train_data_iter = iter(train_loader)
                inputs = next(train_data_iter)
            
            inputs = [v.to(device) for v in inputs]
            triplets = convert_triplets(inputs, args, gen_model, device)
            loss = 0.0
            for tp in triplets:
                loss += compute_ret_loss(
                    ret_model, 
                    criterion,
                    tp
                )

            ret_opt.zero_grad()
            loss.backward()
            if args['world_size'] > 1:
                average_gradients(ret_model)
            torch.nn.utils.clip_grad_norm_(ret_model.parameters(), 2.0)
            ret_opt.step()

            # update learning rate
            ret_scheduler.step_update(cur_step)
            
            # compute moving average of network parameters
            moving_average(ret_model, ret_ema)
            # save model checkpoints
            if mark_flag and (cur_step+1) % args['save_every'] == 0:
                save_checkpoint(ret_model, args['checkpoint_dir'], cur_step+1, "ret")
                save_checkpoint(ret_ema, args['checkpoint_dir'], cur_step+1, "ret_ema")

            # print out log info
            if mark_flag and (cur_step+1) % args['print_every'] == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Epoch [%i/%i], Iteration [%i/%i], " % (elapsed, epoch, args['total_epochs'], cur_step+1, total_steps)
                log += "loss: %.4f, lr: %.6f" % (loss, ret_opt.param_groups[0]['lr'])
                print(log)

            # validation
            if mark_flag and (cur_step+1) % args['valid_every'] == 0:
                correct = validation(
                    test_loader, 
                    args,
                    ret_model, 
                    gen_model,
                    args['margin'], 
                    args['norm_degree'],
                    device
                )
                print("Val correctness: %.4f" % correct.item())

            cur_step += 1

    # save the image embeddings:
    if mark_flag:
        # save the finale models
        save_checkpoint(ret_model, args['checkpoint_dir'], cur_step+1, "ret")
        save_checkpoint(ret_ema, args['checkpoint_dir'], cur_step+1, "ret_ema")
        
        # collect image embeddings
        data_loader = build_ret_loader(
            args['image_dir'],
            args['train_list_path'],
            args['crop_size'],
            args['img_size'],
            args['dataset_name'],
            args['batch_size'], 
            args['num_workers']
        )
        image_embeddings = collect_embeddings(
            data_loader,
            gen_model.eval(),
            ret_model.eval(),
            device
        )

        save_path = os.path.join(args['checkpoint_dir'], 'image_embeddings.pth')
        torch.save(image_embeddings, save_path)
        print('Saving image embeddings into %s...' % save_path)
