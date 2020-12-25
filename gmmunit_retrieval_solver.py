import os
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable
from torchvision import transforms as T

from networks.networks_gmmunit_retrieval import AdaINGen, MsImageDis
from networks.retrieval_networks import RetrievalNet

from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from gmm import gmm_kl_distance
from tools import dist_sampling_split

#------------------------------------------------------------#
#     Solver for using content-attribute disentangling       #
#------------------------------------------------------------#
class GMM_Solver(nn.Module):
    def __init__(self, configs, device=None):
        super(GMM_Solver, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.configs =  configs

        # Initiate the networks
        self.gen = AdaINGen(configs['input_dim'], configs['gen'])  # auto-encoder for domain a
        self.dis = MsImageDis(configs['input_dim'], configs['dis'], self.device)  # discriminator for domain a
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.ret = RetrievalNet(configs['ret'])

        self.print_network(self.dis, 'Dis')
        self.print_network(self.gen, 'Gen')
        self.print_network(self.ret, 'Ret')

        # Network weight initialization
        self.apply(weights_init(configs['init']))
        self.dis.apply(weights_init('gaussian'))
        self.ret.apply(weights_init('gaussian'))

        lr = configs['lr']
        self.lr_policy = configs['lr_policy']

        # fix parameters
        self.freezen_networks()
        # Setup the optimizers
        beta1, beta2 = configs['beta1'], configs['beta2']
        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=configs['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=configs['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, configs)
        self.gen_scheduler = get_scheduler(self.gen_opt, configs)


        # Create Transformation to load all images transformed to RAM (for retrieval)
        self.image_size = configs['image_size']
        self.transform = []
        self.transform.append(T.CenterCrop(configs['crop_size']))
        self.transform.append(T.Resize(self.image_size))
        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T.Compose(self.transform)

        # Load train images embeddings and all train images to RAM
        img_em_data = json.load(open(configs['ret']['img_emb_path']))
        print("Number of image embeddings loaded: {}".format(len(img_em_data)))
        self.img_em = torch.zeros(len(img_em_data), 
            configs['ret']['embed_dim'], dtype=torch.float32).to(device)
        self.img = torch.zeros(len(img_em_data), 3, 
            self.image_size, self.image_size, dtype=torch.float32)
        print("Loading training images and their embeddings")
        for i,(k,v) in enumerate(img_em_data.items()):
            self.img_em[i,:] = torch.from_numpy(np.array(v['embedding']))
            image = Image.open(os.path.join(configs['data_root'], k))
            image = self.transform(image)
            self.img[i,:,:,:] = image
            if i % 10000 == 0:
                print("Loading images and embeddings: {} / {}".format(i,len(img_em_data))) 
        del img_em_data
        print("Images and images embeddings loaded")
        
        self.num_results = configs['ret']['num_results']
        #self.v_dim = configs['v_dim']
        self.c_dim = configs['c_dim']
        self.dist_mode = configs['dist_mode']
        self.num_class = configs['gen']['num_cls']
        self.use_attention = configs['gen']['use_attention']
        self.use_dsn = configs['gen']['use_dsn']
        self.style_dim = self.num_class*self.c_dim

        # fix the noise used in sampling
        display_size = int(configs['display_size'])
        self.display_size = display_size

        self.dataset = configs['dataset']
        self.stddev  = configs['stddev']

        self.criterionL1 = torch.nn.L1Loss()
        self.dist = nn.PairwiseDistance(p=configs['ret']['distance_norm_degree'])

        # Load VGG model if needed
        if 'vgg_w' in configs.keys() and configs['vgg_w'] > 0:
            self.vgg = load_vgg16(configs['vgg_model_path'] + '/models').to(device)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print("The number of parameters: {}".format(num_params))

    def freezen_networks(self):
        self.gen.freezen_params()
        self.ret.freezen_params()

    def update_learning_rate(self):
        if self.lr_policy == 'cosa':
            if self.dis_opt.param_groups[0]['lr'] == self.configs['eta_min'] or \
                self.gen_opt.param_groups[0]['lr'] == self.configs['eta_min']:
                self.configs['step_size'] *= self.configs['t_mult']
                self.dis_scheduler = get_scheduler(self.dis_opt, self.configs)
                self.gen_scheduler = get_scheduler(self.gen_opt, self.configs)

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def recon_criterion(self, x, y):
        return torch.mean(torch.abs(x - y))

    def distance(self, x, y):
        return torch.mean(torch.abs(x-y).sum(dim=1))

    def isometry_constraint(self, z1, z2, rec_z1, rec_z2):
        return torch.abs(self.distance(z1, z2) - self.distance(rec_z1, rec_z2)).mean()

    def mode_seeking_constraint(self, im1, im2, z1, z2, eps=1e-5):
        loss = torch.mean(torch.abs(im1 - im2)) / torch.mean(torch.abs(z1 - z2))
        return 1.0 / (loss + eps)

    def criterion_l1(self, a, z):
        if isinstance(a, list):
            a = torch.cat(a, dim=1)
        if isinstance(z, list):
            z = torch.cat(z, dim=1)
        return self.criterionL1(a, z)

    def style_replace(self, c_src, c_trg, z_src, z_trg):
        mark = c_src==c_trg
        for i in range(c_src.size(0)):
            for j in range(c_src.size(1)):
                if mark[i,j]:
                    z_trg[i, j*self.c_dim:(j+1)*self.c_dim] = z_src[i, j*self.c_dim:(j+1)*self.c_dim].clone()
        return z_trg

    def forward(self, x_src, x_trg=None, c_trg=None):
        cont_src, sty_src_outs = self.gen.encode(x_src)
        if x_trg is not None:
            _, sty_trg_outs = self.gen.encode(x_trg)
            sty_trg = torch.cat(sty_trg_outs[0],dim=1)
        else:
            sty_trg = dist_sampling_split(c_trg, self.c_dim, self.stddev, self.device)
            sty_src = torch.cat(sty_src_outs[0],dim=1)
            c_src = torch.ones(1,self.num_class).float().to(self.device)
            for idx in range(self.num_class):
                if sty_src[0,idx*self.c_dim:(idx+1)*self.c_dim].mean() < 0.0:
                    c_src[0,idx] = -1.0
            sty_trg = self.style_replace(c_src, c_trg, sty_src, sty_trg)
        x_fake, x_fake_att = self.gen.decode(cont_src, sty_trg)
        
        if self.use_attention:
            x_fake = x_fake * x_fake_att  + x_src * (1-x_fake_att) 
        return x_fake

    def gen_update(self, x_src, c_src, label_src, 
        x_trg, c_trg, label_trg, configs):
        self.gen_opt.zero_grad()
        
        # Encode the source image and reconstruct itself
        cont_src, sty_src_prime = self.gen.encode(x_src)
        sty_src = torch.cat(sty_src_prime[0],dim=1)

        retrieved_images_src, _ = self.retrieve_closer_images(cont_src, 
            sty_src, self.num_results)
        retrieved_feats_src = self.gen.encode_retrieved(retrieved_images_src)
        x_src_rec, x_src_rec_att = self.gen.decode(cont_src.detach(), 
            sty_src.detach(), retrieved_feats_src.detach())
        if self.use_attention:
            x_src_rec = x_src_rec*x_src_rec_att + x_src*(1-x_src_rec_att)
        cont_src_rec, sty_src_prime_rec = self.gen.encode(x_src_rec)

        # Encode the target image and sampling from the domain
        #cont_trg, sty_trg_prime = self.gen.encode(x_trg)
        z_trg_rand = dist_sampling_split(c_trg, self.c_dim, self.stddev, self.device)
        #if random.random() > 0.5:
        #    z_trg_rand = self.style_replace(c_src, c_trg, sty_src, z_trg_rand).detach()
        #sty_trg = torch.cat(sty_trg_prime[0],dim=1)

        retrieved_images_trg, _ = self.retrieve_closer_images(cont_src, 
            z_trg_rand, self.num_results)
        retrieved_feats_trg = self.gen.encode_retrieved(retrieved_images_trg)
        x_fake1, x_fake1_att = self.gen.decode(cont_src.detach(), 
            z_trg_rand, retrieved_feats_trg.detach())

        #retrieved_images_trg, _ = self.retrieve_closer_images(cont_src, 
        #    sty_trg, self.num_results)
        #retrieved_feats_trg = self.gen.encode_retrieved(retrieved_images_trg)
        #x_fake2, x_fake2_att = self.gen.decode(cont_src.detach(), 
        #    sty_trg.detach(), retrieved_feats_trg.detach())
        if self.use_attention:
            x_fake1 = x_fake1*x_fake1_att + x_src*(1-x_fake1_att)
            #x_fake2 = x_fake2*x_fake2_att + x_src*(1-x_fake2_att)

        # Encode the fake image again
        cont_fake1, sty_fake_outs1 = self.gen.encode(x_fake1)
        #cont_fake2, sty_fake_outs2 = self.gen.encode(x_fake2)

        # decode again (if needed)
        if configs['recon_x_cyc_w'] > 0:
            x_cycle1, x_cycle1_att = self.gen.decode(cont_fake1.detach(), 
                sty_src.detach(), retrieved_feats_src.detach())
            #x_cycle2, x_cycle2_att = self.gen.decode(cont_fake2.detach(), 
            #    sty_src.detach(), retrieved_feats_src.detach())
            if self.use_attention:
                x_cycle1 = x_cycle1*x_cycle1_att + x_src*(1-x_cycle1_att)
                #x_cycle2 = x_cycle2*x_cycle2_att + x_src*(1-x_cycle2_att)

        # reconstruction loss
        self.loss_gen_recon_x  = self.recon_criterion(x_src_rec, x_src)
        self.loss_gen_recon_c  = self.recon_criterion(cont_src_rec, cont_src)
        self.loss_gen_recon_c1 = self.recon_criterion(cont_fake1, cont_src)
        #self.loss_gen_recon_c2 = self.recon_criterion(cont_fake2, cont_src)
        self.loss_gen_recon_s  = self.criterion_l1(sty_src_prime_rec[0], sty_src)
        self.loss_gen_recon_s1 = self.criterion_l1(sty_fake_outs1[0], z_trg_rand)
        #self.loss_gen_recon_s2 = self.criterion_l1(sty_fake_outs2[0], sty_trg)

        self.loss_gen_cycle1, self.loss_gen_cycle2 = 0., 0.
        if configs['recon_x_cyc_w'] > 0:
            self.loss_gen_cycle1 = self.recon_criterion(x_cycle1, x_src)
            #self.loss_gen_cycle2 = self.recon_criterion(x_cycle2, x_src)

        # GAN loss
        self.loss_gen_adv1 = self.dis.calc_gen_loss(x_fake1, label_trg, configs['gan_w'], configs['cls_w'])
        #self.loss_gen_adv2 = self.dis.calc_gen_loss(x_fake2, label_trg, configs['gan_w'], configs['cls_w'])
        
        # domain-invariant perceptual loss
        self.loss_gen_vgg1, self.loss_gen_vgg2 = 0.0, 0.0
        if configs['recon_x_cyc_w'] > 0 and configs['vgg_w'] > 0:
            self.loss_gen_vgg1 = self.compute_vgg_loss(self.vgg, x_src, x_cycle1)
            #self.loss_gen_vgg2 = self.compute_vgg_loss(self.vgg, x_src, x_cycle2)

        # total loss
        self.loss_gen_total = self.loss_gen_adv1 + \
                              configs['recon_x_w'] * self.loss_gen_recon_x + \
                              configs['recon_c_w'] * self.loss_gen_recon_c + \
                              configs['recon_c_w'] * self.loss_gen_recon_c1 + \
                              configs['recon_s_w'] * self.loss_gen_recon_s + \
                              configs['recon_s_w'] * self.loss_gen_recon_s1 + \
                              configs['recon_x_cyc_w'] * self.loss_gen_cycle1 + \
                              configs['vgg_w'] * self.loss_gen_vgg1
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.device)
        target_vgg = vgg_preprocess(target, self.device)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_src, c_trg):
        self.eval()
        z_trg_random1 = dist_sampling_split(c_trg, self.c_dim, self.stddev, self.device)
        z_trg_random2 = dist_sampling_split(c_trg, self.c_dim, self.stddev, self.device)

        x_src_rec, x_fake1, x_fake2, att_rec, att_fake1, att_fake2 = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(x_src.size(0)):
                cont_src, sty_src_outs = self.gen.encode(x_src[i:i+1])
                sty_src = torch.cat(sty_src_outs[0], dim=1)
                c_src = torch.ones(1, self.num_class).float().to(self.device)
                for idx in range(self.num_class):
                    if sty_src[0,idx*self.c_dim:(idx+1)*self.c_dim].mean() < 0.0:
                        c_src[0,idx] = -1.0

                retrieved_images_src, _ = self.retrieve_closer_images(cont_src, 
                    sty_src, self.num_results)
                retrieved_feats_src = self.gen.encode_retrieved(retrieved_images_src)
                x_real_rec, x_real_rec_att = self.gen.decode(cont_src, 
                    sty_src, retrieved_feats_src)

                z_rand1 = z_trg_random1[i:i+1]
                #z_rand1 = self.style_replace(c_src, c_trg, sty_src, z_rand1)
                retrieved_images_trg, _ = self.retrieve_closer_images(cont_src, 
                    z_rand1, self.num_results)
                retrieved_feats_trg = self.gen.encode_retrieved(retrieved_images_trg)
                x_tgt1, x_tgt1_att = self.gen.decode(cont_src, z_rand1, retrieved_feats_trg)

                z_rand2 = z_trg_random2[i:i+1]
                #z_rand2 = self.style_replace(c_src, c_trg, sty_src, z_rand2)
                retrieved_images_trg, _ = self.retrieve_closer_images(cont_src, 
                    z_rand2, self.num_results)
                retrieved_feats_trg = self.gen.encode_retrieved(retrieved_images_trg)
                x_tgt2, x_tgt2_att = self.gen.decode(cont_src, z_rand2, retrieved_feats_trg)
                if self.use_attention:
                    x_tgt1 = x_tgt1*x_tgt1_att + x_src[i:i+1]*(1-x_tgt1_att)
                    x_tgt2 = x_tgt2*x_tgt2_att + x_src[i:i+1]*(1-x_tgt2_att)
                    x_real_rec = x_real_rec*x_real_rec_att + x_src[i:i+1]*(1-x_real_rec_att)
                    att_rec.append(torch.cat([x_real_rec_att,x_real_rec_att,x_real_rec_att],dim=1))
                    att_fake1.append(torch.cat([x_tgt1_att,x_tgt1_att,x_tgt1_att],dim=1))
                    att_fake2.append(torch.cat([x_tgt2_att,x_tgt2_att,x_tgt2_att],dim=1))
                x_fake1.append(x_tgt1)
                x_fake2.append(x_tgt2)
                x_src_rec.append(x_real_rec)
        x_src_rec = torch.cat(x_src_rec)
        x_fake1, x_fake2 = torch.cat(x_fake1), torch.cat(x_fake2)
        outputs = [x_src, x_src_rec, x_fake1, x_fake2]
        if self.use_attention:
            att_fake1 = (torch.cat(att_fake1)-0.5)/0.5 
            att_fake2 = (torch.cat(att_fake2)-0.5)/0.5
            outputs += [att_fake1, att_fake2]
        self.train()
        return outputs

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def dis_update(self, x_src, c_src, label_src, 
        x_trg, c_trg, label_trg, configs):
        self.dis_opt.zero_grad()
        cont_src, sty_src_outs = self.gen.encode(x_src)
        sty_src = torch.cat(sty_src_outs[0],dim=1)

        z_trg_rand = dist_sampling_split(c_trg, self.c_dim, self.stddev, self.device)
        #if random.random() > 0.5:
        #    z_trg_rand = self.style_replace(c_src, c_trg, sty_src, z_trg_rand).detach()
        #cont_trg, sty_trg_outs = self.gen.encode(x_trg)
        #sty_trg = torch.cat(sty_trg_outs[0], dim=1)

        retrieved_images_trg, _ = self.retrieve_closer_images(cont_src, 
            z_trg_rand, self.num_results)
        retrieved_feats_trg = self.gen.encode_retrieved(retrieved_images_trg)
        x_fake1, x_fake_att1 = self.gen.decode(cont_src, z_trg_rand, retrieved_feats_trg)

        #retrieved_images_trg, _ = self.retrieve_closer_images(cont_src, 
        #    sty_trg, self.num_results)
        #retrieved_feats_trg = self.gen.encode_retrieved(retrieved_images_trg)
        #x_fake2, x_fake_att2 = self.gen.decode(cont_src, sty_trg, retrieved_feats_trg)
        
        if self.use_attention:
            x_fake1 = x_fake1*x_fake_att1 + x_src*(1-x_fake_att1)
            #x_fake2 = x_fake2*x_fake_att2 + x_src*(1-x_fake_att2)

        self.loss_dis1 = self.dis.calc_dis_loss(x_fake1, x_src, label_src, configs['gan_w'], configs['cls_w'])
        #self.loss_dis2 = self.dis.calc_dis_loss(x_fake2, x_trg, label_trg, configs['gan_w'], configs['cls_w'])

        # Compute loss for gradient penalty.
        self.loss_gp = 0.0
        if configs['gp_w'] > 0.0:
            alpha = torch.rand(x_src.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_src.data + (1 - alpha) * x_fake1.data).requires_grad_(True)
            out_src, _ = self.dis.forward(x_hat, False)[0]
            self.loss_gp += self.gradient_penalty(out_src, x_hat) * configs['gp_w']

        #self.loss_dis_all = self.loss_dis1 + self.loss_dis2 + self.loss_gp
        self.loss_dis_all = self.loss_dis1 + self.loss_gp
        self.loss_dis_all.backward()
        self.dis_opt.step()

    def resume(self, checkpoint_dir, configs):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name, map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name, map_location=lambda storage, loc: storage)
        self.dis.load_state_dict(state_dict['b'])
        # Load optimizers
        #state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'), map_location=lambda storage, loc: storage)
        #self.dis_opt.load_state_dict(state_dict['dis'])
        #self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, configs, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, configs, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen.state_dict()}, gen_name)
        torch.save({'b': self.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def load_ret_checkpoint(self, checkpoint_dir):
        # Load generators
        state_dict = torch.load(checkpoint_dir, map_location=lambda storage, loc: storage)
        self.ret.load_state_dict(state_dict['a'])
        print('Retrieval model checkpoint loaded.')

    def init_network(self, gen_path, dis_path):
        """In order to tuning the models with CAMs"""
        gen_dict = torch.load(gen_path, map_location=lambda storage, loc: storage)['a']
        dis_dict = torch.load(dis_path, map_location=lambda storage, loc: storage)['b']
        dis_state_dict = self.dis.state_dict()
        gen_state_dict = self.gen.state_dict()
        
        for key in dis_state_dict:
            if key in dis_dict:
                dis_state_dict[key] = dis_dict[key]
        self.dis.load_state_dict(dis_dict)

        for key in gen_state_dict:
            if key in gen_dict:
                gen_state_dict[key] = gen_dict[key]
        self.gen.load_state_dict(gen_state_dict)

        print("Initial model loaded...")


    def retrieve_closer_images(self, i_con, i_att, num_results):
        # Get query embedding
        with torch.no_grad():
            # Compute query embedding
            query_em = self.ret(i_con, i_att)
            # Get nearest images
            distances = self.dist(self.img_em, query_em)
            distances = distances.sort(descending=False)
            results_indices = distances[1][0:num_results]
            results_distances = distances[0][0:num_results]
            retrieved_images = torch.zeros(num_results, 3, self.image_size, self.image_size, dtype=torch.float32)
            retrieved_images = self.img[results_indices,:,:,:]
            retrieved_images = retrieved_images.to(self.device)
        return retrieved_images, results_distances
