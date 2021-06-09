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

from .losses import (
    calc_d_loss,
    calc_g_loss,
    criterion_vgg,
    criterion_l1,
    criterion_kl
)

from .models import (
    AdaINGen,
    MsImageDis
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
    build_perceptual_nets,
    style_replace
)

from data import build_loader, build_dataset
from .optims import (
    build_scheduler,
    build_optimizer
)

import random
random.seed(1234)

def average_gradients(model):
    """ Gradient averaging. """
    size = float(distributed.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            distributed.all_reduce(param.grad.data, op=distributed.ReduceOp.SUM)
            param.grad.data /= size

def compute_g_loss(
    dis_model, 
    gen_model,
    args,
    src_img,
    trg_img,
    src_lab,
    trg_lab,
    gmm_src,
    gmm_trg,
    use_attention,
    vgg=None,
    inst_norm=None,
    device=None
):
    batch_size    = args['batch_size']
    attr_dim      = args['attr_dim']
    dataset_name  = args['dataset_name']
    lambda_adv    = args['lambda_adv']
    lambda_cls    = args['lambda_cls']
    lambda_gp     = args['lambda_gp']
    lambda_kl     = args['lambda_kl']
    gan_type      = args['gan_type']

    # Encode the source image and self-reconstruction
    cont_src, sty_src = gen_model.encode(src_img)
    src_img_rec, src_att_rec = gen_model.decode(
        cont_src, torch.cat(sty_src,dim=1))
    if use_attention:
        src_img_rec = src_img_rec*src_att_rec + src_img*(1-src_att_rec)
    loss_rec = criterion_l1(src_img, src_img_rec) 

    # Forward translation to target attributes
    if trg_img is None:
        z_trg = dist_sampling_split(gmm_trg, attr_dim, device)
        if random.random() > 0.5:
            z_trg = style_replace(
                gmm_src, 
                gmm_trg, 
                torch.stack(sty_src, dim=1),  # [N, S*D]
                z_trg.view(batch_size, -1, attr_dim)
            ).view(batch_size, -1)    # [N, S*D]
        img_fake, att_fake = gen_model.decode(cont_src, z_trg)
    else:
        _, sty_trg = gen_model.encode(trg_img)
        img_fake, att_fake = gen_model.decode(
            cont_src, torch.cat(sty_trg,dim=1))
    if use_attention:
        img_fake = img_fake*att_fake + src_img*(1-att_fake)
    loss_gen = calc_g_loss(
        dis_model,
        img_fake,
        trg_lab,
        lambda_adv,
        lambda_cls,
        gan_type,
        dataset_name
    )

    # Encode the fake image again
    cont_fake, sty_fake = gen_model.encode(img_fake)
    loss_cont_rec = criterion_l1(cont_fake, cont_src)

    # cycle consistency
    cyc_img, cyc_att = gen_model.decode(
        cont_fake, torch.cat(sty_src,dim=1))
    if use_attention:
        cyc_img = cyc_img*cyc_att + src_img*(1-cyc_att)
    loss_cyc = criterion_l1(cyc_img, src_img)
    
    # cycle encode style
    _, sty_cyc = gen_model.encode(cyc_img)
    loss_cyc_sty = criterion_l1(
        torch.cat(sty_src,dim=1), 
        torch.cat(sty_cyc,dim=1))

    # KL loss
    loss_kl = criterion_kl(torch.stack(sty_src,dim=1), gmm_src.unsqueeze(2))

    gen_loss = loss_gen + \
               loss_rec * args['lambda_recx'] + \
               loss_cont_rec * args['lambda_rec'] + \
               loss_cyc * args['lambda_cyc'] + \
               loss_cyc_sty * args['lambda_rec'] + \
               loss_kl * args['lambda_kl'] 

    g_losses = Munch(rec=loss_rec.item(),
                     gen=loss_gen.item(),
                     cont_rec=loss_cont_rec.item(),
                     cyc=loss_cyc.item(),
                     cyc_sty=loss_cyc_sty.item(),
                     kl=loss_kl.item())

    # VGG perceptual loss
    if args['lambda_vgg'] > 0:
        loss_vgg = criterion_vgg(src_img, cyc_img, vgg, inst_norm, device)
        gen_loss += loss_vgg * args['lambda_vgg']
        g_losses.vgg = loss_vgg.item()
    return gen_loss, g_losses

def compute_d_loss(
    dis_model, 
    gen_model,
    args,
    src_img,
    trg_img,
    src_lab,
    gmm_src,
    gmm_trg,
    use_attention, 
    device
):
    batch_size    = args['batch_size']
    attr_dim      = args['attr_dim']
    dataset_name  = args['dataset_name']
    lambda_adv    = args['lambda_adv']
    lambda_cls    = args['lambda_cls']
    lambda_gp     = args['lambda_gp']
    gan_type      = args['gan_type']

    cont_src, sty_src = gen_model.encode(src_img)
    if trg_img is None:
        z_trg = dist_sampling_split(gmm_trg, attr_dim, device)
        if random.random() > 0.5:
            z_trg = style_replace(
                gmm_src, 
                gmm_trg, 
                torch.stack(sty_src, dim=1),  # [N, S*D]
                z_trg.view(batch_size, -1, attr_dim)
            ).view(batch_size, -1)    # [N, S*D]
        img_fake, att_fake = gen_model.decode(cont_src, z_trg)
    else:
        _, sty_trg = gen_model.encode(trg_img)
        img_fake, att_fake = gen_model.decode(
            cont_src, torch.cat(sty_trg,dim=1))

    if use_attention:
        img_fake = img_fake*att_fake + src_img*(1-att_fake)

    dis_loss = calc_d_loss(
        dis_model,
        src_img,
        img_fake,
        src_lab,
        lambda_adv,
        lambda_cls,
        lambda_gp,
        gan_type,
        dataset_name,
        device
    )
    d_losses = Munch(dis=dis_loss.item())
    return dis_loss, d_losses

def gmmunit_run(args, local_rank):
    print(distributed.get_rank())
    if args['world_size'] > 1:
        torch.manual_seed(1234+distributed.get_rank())
        random.seed(5678+distributed.get_rank())
    device = torch.device('cuda', local_rank)
    mark_flag = args['world_size']==1 or distributed.get_rank() ==0

    # build models
    print("Build and initilize models ...")
    gen_model = AdaINGen(args['gen']).to(device)
    dis_model = MsImageDis(args['dis']).to(device)
    print_network(gen_model, 'GEN')
    print_network(dis_model, 'DIS')
    gen_model.apply(weights_init('kaiming'))
    dis_model.apply(weights_init('gaussian'))

    # resume training if necessary
    load_checkpoint(gen_model, args['checkpoint_dir'], args['resume_iter'], suffix='gen_ema')
    load_checkpoint(dis_model, args['checkpoint_dir'], args['resume_iter'], suffix='dis_ema')
    gen_ema = copy.deepcopy(gen_model)
    dis_ema = copy.deepcopy(dis_model)

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
    gen_opt = build_optimizer(gen_model, base_lr=args['lr'])
    dis_opt = build_optimizer(dis_model, base_lr=args['lr'])
    n_iter_per_epoch = len(train_loader)
    gen_scheduler = build_scheduler(
        gen_opt, 
        args['total_epochs'], 
        n_iter_per_epoch,
        scheduler_type=args['scheduler_type'],
        warmup_epochs=args['warmup_epochs']
    )
    dis_scheduler = build_scheduler(
        dis_opt, 
        args['total_epochs'], 
        n_iter_per_epoch,
        scheduler_type=args['scheduler_type'],
        warmup_epochs=args['warmup_epochs']
    )

    total_steps = args['total_epochs'] * n_iter_per_epoch
    start_epoch = int(args['resume_iter'] / n_iter_per_epoch) if args['resume_iter'] > 0 else 0

    # build perceptual nets
    vgg, inst_norm = None, None
    if args['lambda_vgg'] > 0: 
        vgg, inst_norm = build_perceptual_nets(args)
        vgg.to(device)

    attention_warmup_epochs = args['attention_warmup_epochs']
    kl_warmup_epochs = args['kl_warmup_epochs']
    use_attention    = args['gen']['use_attention']
    init_lambda_kl   = args['lambda_kl']

    cur_step = args['resume_iter'] if args['resume_iter'] > 0 else 0
    print("Start Epoch: {}, Start Step: {} ...".format(start_epoch, cur_step))
    for epoch in range(start_epoch, args['total_epochs']):
        train_loader.sampler.set_epoch(epoch)
        train_data_iter = iter(train_loader)
        test_data_iter  = iter(test_loader)
        
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
            src_img, trg_img, src_lab, trg_lab = inputs

            gmm_src = assign_gmm_componet(src_lab, num_cls=args['num_cls'], mode=args['dataset_name']).to(device)
            gmm_trg = assign_gmm_componet(trg_lab, num_cls=args['num_cls'], mode=args['dataset_name']).to(device)

            # if using KL, gradually increasing the loss weight
            if init_lambda_kl > 0:
                if epoch > kl_warmup_epochs:
                    args['lambda_kl'] = (epoch - kl_warmup_epochs) * init_lambda_kl / (args['total_epochs'] - kl_warmup_epochs)
                else:
                    args['lambda_kl'] = 0

            # train the discriminator
            d_loss, d_losses_lat = compute_d_loss(
                dis_model, 
                gen_model,
                args,
                src_img,
                None,     # sampling in latent space
                src_lab,
                gmm_src,
                gmm_trg,
                use_attention and epoch > attention_warmup_epochs,
                device
            )
            dis_opt.zero_grad()
            d_loss.backward()
            if args['world_size'] > 1:
                average_gradients(dis_model)
            torch.nn.utils.clip_grad_norm_(dis_model.parameters(), 2.0)
            dis_opt.step()
            
            d_loss, d_losses_ref = compute_d_loss(
                dis_model, 
                gen_model,
                args,
                src_img,
                trg_img,  # sampling real image
                src_lab,
                gmm_src,
                gmm_trg,
                use_attention and epoch > attention_warmup_epochs,
                device
            )
            dis_opt.zero_grad()
            d_loss.backward()
            if args['world_size'] > 1:
                average_gradients(dis_model)
            torch.nn.utils.clip_grad_norm_(dis_model.parameters(), 2.0)
            dis_opt.step()

            # train the generator
            g_loss, g_losses_lat = compute_g_loss(
                dis_model, 
                gen_model,
                args,
                src_img,
                None,     # sampling in latent space
                src_lab,
                trg_lab,
                gmm_src,
                gmm_trg,
                use_attention and epoch > attention_warmup_epochs,
                vgg,
                inst_norm,
                device
            )
            gen_opt.zero_grad()
            g_loss.backward()
            if args['world_size'] > 1:
                average_gradients(gen_model)
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 2.0)
            gen_opt.step()

            g_loss, g_losses_ref = compute_g_loss(
                dis_model, 
                gen_model,
                args,
                src_img,
                trg_img,  # sampling real image
                src_lab,
                trg_lab,
                gmm_src,
                gmm_trg,
                use_attention and epoch > attention_warmup_epochs,
                vgg,
                inst_norm,
                device
            )
            gen_opt.zero_grad()
            g_loss.backward()
            if args['world_size'] > 1:
                average_gradients(gen_model)
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 2.0)
            gen_opt.step()
            
            # update learning rate
            dis_scheduler.step_update(cur_step)
            gen_scheduler.step_update(cur_step)

            # compute moving average of network parameters
            moving_average(gen_model, gen_ema)
            moving_average(dis_model, dis_ema)

            # save model checkpoints
            if mark_flag and (cur_step+1) % args['save_every'] == 0:
                save_checkpoint(gen_model, args['checkpoint_dir'], cur_step+1, "gen")
                save_checkpoint(dis_model, args['checkpoint_dir'], cur_step+1, "dis")
                save_checkpoint(gen_ema, args['checkpoint_dir'], cur_step+1, "gen_ema")
                save_checkpoint(dis_ema, args['checkpoint_dir'], cur_step+1, "dis_ema")

            # print out log info
            if mark_flag and (cur_step+1) % args['print_every'] == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Epoch [%i/%i], Iteration [%i/%i], " % (elapsed, epoch, args['total_epochs'], cur_step+1, total_steps)

                all_losses = dict()
                for loss, prefix in zip(
                    [d_losses_lat, d_losses_ref, g_losses_lat, g_losses_ref], 
                    ['D/lat_', 'D/ref_', 'G/lat_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses["G/lr"] = gen_opt.param_groups[0]['lr']
                all_losses["D/lr"] = dis_opt.param_groups[0]['lr']
                all_losses["kl/w"] = args['lambda_kl']
                log += ' '.join(['%s: [%.4f]' % (key, value) if 'lr' not in key else '%s: [%.6f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if mark_flag and (cur_step+1) % args['sample_every'] == 0:
                if not os.path.exists(args['sample_dir']):
                    os.mkdir(args['sample_dir'])
                # save training images
                z_trg = dist_sampling_split(gmm_trg, args['attr_dim'], device)
                save_name = os.path.join(args['sample_dir'], '{:08d}_train.jpg'.format(cur_step+1))
                debug_image(gen_model, src_img, trg_img, z_trg, save_name, use_attention and epoch > attention_warmup_epochs)

                # save testing images
                try:
                    src_img_val, trg_img_val, src_lab_val, trg_lab_val = next(test_data_iter)
                except:
                    test_data_iter = iter(test_loader)
                    src_img_val, trg_img_val, src_lab_val, trg_lab_val = next(test_data_iter)
                src_img_val = src_img_val.to(device)
                trg_img_val = trg_img_val.to(device)
                src_lab_val = src_lab_val.to(device)
                trg_lab_val = trg_lab_val.to(device)
                gmm_trg_val = assign_gmm_componet(trg_lab_val, num_cls=args['num_cls'], mode=args['dataset_name']).to(device)
                z_trg = dist_sampling_split(gmm_trg_val, args['attr_dim'], device)
                save_name   = os.path.join(args['sample_dir'], '{:08d}_test.jpg'.format(cur_step+1))
                debug_image(gen_model, src_img_val, trg_img_val, z_trg, save_name, use_attention and epoch > attention_warmup_epochs)

            cur_step += 1


