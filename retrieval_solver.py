import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable

from networks.networks import AdaINGen
from networks.networks_gmmunit_retrieval import AdaINGenRaw
from networks.networks_stargan_retrieval import Generator
from networks.retrieval_networks import RetrievalNet
from utils import weights_init, get_model_list, get_scheduler
from data_ios.dwcgan_data.vocab import Vocab

class GMM_Solver(nn.Module):
    def __init__(self, configs, device=None):
        super(GMM_Solver, self).__init__()
        self.device = device if device is not None else torch.device('cpu')

        # Initiate the networks
        self.gen = AdaINGenRaw(configs['input_dim'], configs['gen'])
        self.gen.eval()
        for param in self.gen.parameters():
            param.requires_grad = False

        self.use_attention = configs['gen']['use_attention']

    def forward(self, x):
        pass

    def gen_decode(self, x_real, content, style):
        x_fake, x_fake_att = self.gen.decode(content, style)
        if self.use_attention:
            x_fake = x_fake * x_fake_att  + x_real * (1-x_fake_att) 
        return x_fake

    def gen_encode(self, x_real):
        content, style_outs = self.gen.encode(x_real)
        return content, torch.cat(style_outs[0],dim=1)

    def gen_rec(self, x_real):
        content_real, style_outs = self.gen.encode(x_real)
        style_real = torch.cat(style_outs[0],dim=1)
        x_real_rec, x_real_rec_att = self.gen.decode(content_real, style_real)
        if self.use_attention:
            x_real_rec = x_real_rec*x_real_rec_att + x_real*(1-x_real_rec_att)
        return x_real_rec

    def initial_network(self, model_path):
        gen_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['a']
        self.gen.load_state_dict(gen_dict)


class DWC_Solver(nn.Module):
    def __init__(self, configs, device=None, pretrained_embed=None):
        super(DWC_Solver, self).__init__()
        self.device = device if device is not None else torch.device('cpu')

        self.vocab = Vocab(dataset=configs['dataset'])
        # Initiate the networks
        self.gen = AdaINGen(configs['input_dim'], self.vocab, configs['gen'], 
            pretrained_embed=pretrained_embed)
        self.gen.eval()
        for param in self.gen.parameters():
            param.requires_grad = False

        self.use_attention = configs['gen']['use_attention']

    def forward(self, x):
        pass

    def gen_decode(self, x_real, content, style):
        x_fake, x_fake_att = self.gen.decode(content, style)
        if self.use_attention:
            x_fake = x_fake * x_fake_att  + x_real * (1-x_fake_att) 
        return x_fake

    def gen_encode(self, x_real):
        content, style_src, _ = self.gen.encode(x_real)
        return content, torch.cat(style_src,dim=1)

    def gen_rec(self, x_real):
        content_real, style_real, _ = self.gen.encode(x_real)
        x_real_rec, x_real_rec_att = self.gen.decode(content_real, 
            torch.cat(style_real,dim=1))
        if self.use_attention:
            x_real_rec = x_real_rec*x_real_rec_att + x_real*(1-x_real_rec_att)
        return x_real_rec

    def initial_network(self, model_path):
        gen_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['a']
        self.gen.load_state_dict(gen_dict)


class StarGAN_Solver(nn.Module):
    def __init__(self, configs, device=None):
        super(StarGAN_Solver, self).__init__()
        self.device = device if device is not None else torch.device('cpu')

        # Initiate the networks
        self.gen = Generator(
            configs['dim'], 
            configs['num_cls'], 
            configs['n_res'],
            configs['use_retrieval'])
        self.gen.eval()
        for param in self.gen.parameters():
            param.requires_grad = False

    def forward(self, x):
        pass

    def gen_decode(self, x, c):
        return self.gen.decode(x, c)

    def gen_encode(self, x):
        return self.gen.enc(x)

    def initial_network(self, model_path):
        gen_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(gen_dict)

class Retrieval_Solver(nn.Module):
    def __init__(self, configs, device=None):
        super(Retrieval_Solver, self).__init__()
        lr = configs['lr']
        self.device = device if device is not None else torch.device('cpu')
        #self.vocab = Vocab(dataset=configs['dataset'])

        # Initiate the networks
        self.ret = RetrievalNet(configs['ret'])
        self.print_network(self.ret, "Retrieval Network")

        # Loss
        self.criterion = torch.nn.TripletMarginLoss(margin=configs['margin'], p=configs['distance_norm_degree'])
        self.criterion_eval = torch.nn.TripletMarginLoss(margin=configs['margin'], p=configs['distance_norm_degree'])

        self.margin = configs['margin']
        self.distance_norm_degree = configs['distance_norm_degree']
        #self.v_dim = configs['v_dim']
        #self.c_dim = configs['c_dim']
        #self.style_dim = self.v_dim*self.c_dim
        #self.use_attention = configs['gen']['use_attention']
        self.dataset = configs['dataset']

        # Setup the optimizer
        params = list(self.ret.parameters())
        self.opt = torch.optim.SGD([p for p in params if p.requires_grad],
                                        lr=lr, momentum=configs['momentum'], weight_decay=configs['weight_decay'])
        self.scheduler = get_scheduler(self.opt, configs)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("The number of parameters in {}: {}".format(name, num_params))

    def ret_update(self, a_con, a_att, p_con, p_att, n_con, n_att):
        self.train()
        self.opt.zero_grad()
        a = self.ret(a_con, a_att)
        p = self.ret(p_con, p_att)
        n = self.ret(n_con, n_att)

        # Check if triplet is already correct (not used for the loss, just for monitoring)
        correct = torch.zeros([1], dtype=torch.int32).to(self.device)
        dist_a_p = F.pairwise_distance(a, p, p=self.distance_norm_degree)
        dist_a_n = F.pairwise_distance(a, n, p=self.distance_norm_degree)
        for i in range(0,len(dist_a_p)):
            if (dist_a_n[i] - dist_a_p[i]) > self.margin:
                correct[0] += 1
        self.triplet_loss = self.criterion(a, p, n)
        self.triplet_loss.backward()
        self.opt.step()
        return self.triplet_loss, correct

    def ret_eval(self, a_con, a_att, p_con, p_att, n_con, n_att):
        self.eval()
        with torch.no_grad():
            a = self.ret(a_con, a_att)
            p = self.ret(p_con, p_att)
            n = self.ret(n_con, n_att)

            # Check if triplet is already correct (not used for the loss, just for monitoring)
            correct = torch.zeros([1], dtype=torch.int32).to(self.device)
            dist_a_p = F.pairwise_distance(a, p, p=self.distance_norm_degree)
            dist_a_n = F.pairwise_distance(a, n, p=self.distance_norm_degree)
            for i in range(0,len(dist_a_p)):
                if (dist_a_n[i] - dist_a_p[i]) > self.margin:
                    correct[0] += 1
            self.triplet_loss_eval = self.criterion_eval(a, p, n)
        self.train()
        return self.triplet_loss_eval, correct

    def ret_embedding(self, i_con, i_att):
        self.eval()
        with torch.no_grad():
            embed = self.ret(i_con, i_att)
        self.train()
        return embed
    
    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def resume(self, checkpoint_dir, configs):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "ret")
        state_dict = torch.load(last_model_name, map_location=lambda storage, loc: storage)
        self.ret.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-11:-3])
        # Load optimizer
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'), map_location=lambda storage, loc: storage)
        self.opt.load_state_dict(state_dict['opt'])
        # Reinitilize schedulers
        self.scheduler = get_scheduler(self.opt, configs, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def load_retnet(self, model_path):
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.ret.load_state_dict(state_dict['a'])
        print('Retrieval model checkpoint loaded.')

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        ret_name = os.path.join(snapshot_dir, 'ret_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.ret.state_dict()}, ret_name)
        torch.save({'opt': self.opt.state_dict()}, opt_name)