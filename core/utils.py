# --------------------------------------------------------
# RG-UNIT
# Written by Yahui Liu (yahui.liu@unitn.it)
# --------------------------------------------------------

import os
import yaml
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions as tdist
import torchvision.utils as vutils

from .basic_modules import VGG16

def get_config(config_path):
    with open(config_path, 'r') as fin:
        return yaml.load(fin, Loader=yaml.FullLoader)

# ----------- data ----------- #
def load_image(image_dir, file_path, transform=None):
    img = Image.open(os.path.join(image_dir, file_path)).convert("RGB")
    img = transform(img).unsqueeze(0) # [1, 3, H, W]
    return img

# ----------- gmm ----------- #

def dist_sampling_split(mu, attr_dim=8, device=None):
    cov = torch.ones(mu.size()).to(device) * 0.5
    norm = tdist.Normal(mu, cov)
    sampling = norm.sample((1, attr_dim))
    z_rand = sampling.transpose(2,1).transpose(3,2).contiguous().view(mu.size(0),-1)
    return z_rand

def label2onehot(x, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = x.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), x.long()] = 1
    return out

def assign_gmm_componet(x, num_cls, mode='CelebA'):
    if mode in ['CelebA']:
        gmm_comp = x.clone()
    else: # mode in ['Cat2Dog']
        gmm_comp = label2onehot(x, num_cls)
    gmm_comp = gmm_comp*2.0-1.0
    return gmm_comp


# ----------- training tools ----------- #

def moving_average(model_src, model_ema, beta=0.999):
    for param_src, param_ema in zip(model_src.parameters(), model_ema.parameters()):
        param_ema.data = torch.lerp(param_src.data, param_ema.data, beta)

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else: # init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("{}, Number of parameters: {}".format(name, num_params))


# ----------- save and load ----------- #

def save_checkpoint(model, checkpoint_dir, step, suffix="gen"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    fname = os.path.join(checkpoint_dir, "{:08d}_{}.pth".format(step, suffix))
    print('Saving checkpoint into %s...' % fname)
    torch.save(model.state_dict(), fname)


#def load_checkpoint(model, checkpoint_dir, resume_iter=-1, suffix='gen_ema'):
#    if resume_iter > 0:
#        model_resume_path = os.path.join(checkpoint_dir, "{:08d}_{}.pth".format(resume_iter, suffix))
#        load_pretrained_model(model, model_resume_path)

def load_checkpoint(model, checkpoint_dir, resume_iter=-1, suffix='gen_ema'):
    if resume_iter > 0:
        model_path = os.path.join(checkpoint_dir, "{:08d}_{}.pth".format(resume_iter, suffix))
        state_dict = torch.load(model_path, map_location='cpu')
        print("Loading checkpoint from {}...".format(model_path))

        current_state_dict = model.state_dict()
        for name in current_state_dict:
            if name in state_dict:
                current_state_dict[name] = state_dict[name]
        model.load_state_dict(current_state_dict)


def load_pretrained(pretrained_path):
    assert os.path.exists(pretrained_path), "No such file or folder: {} ...".format(pretrained_path)
    return torch.load(pretrained_path, map_location='cpu')

def load_pretrained_model(model, pretrained_path):
    # strict loading 
    state_dict = load_pretrained(pretrained_path)
    model.load_state_dict(state_dict)
    print("Successfully load pretrained model from {} ...".format(pretrained_path))

def load_pretrained_gmmunit_for_ret(model, pretrained_path):
    # not strict loading 
    state_dict = load_pretrained(pretrained_path)
    current_state_dict = model.state_dict()
    for name in state_dict:
        if name in current_state_dict:
            current_state_dict[name] = state_dict[name]
    model.load_state_dict(current_state_dict)
    print("Successfully load pretrained model from {} ...".format(pretrained_path))

# ----------- perceptual model ----------- #

def build_perceptual_nets(args):
    instancenorm = nn.InstanceNorm2d(512, affine=False) 
    vgg = VGG16(model_path=args['vgg_model_path']).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg, instancenorm

# ----------- debug images ----------- #

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

@torch.no_grad()
def debug_image(
	g_model, 
	src_img, 
	trg_img, 
	z_trg, 
	fname,
	use_attention
):
    g_model.eval()

    bsz = src_img.size(0)
    outputs = [src_img]
    # Encode the source image and self-reconstruction
    cont_src, sty_src = g_model.encode(src_img)
    src_img_rec, src_att_rec = g_model.decode(
        cont_src, torch.cat(sty_src[0],dim=1))
    if use_attention:
        src_img_rec = src_img_rec*src_att_rec + src_img*(1-src_att_rec)
    outputs += [src_img_rec]

    outputs += [trg_img]
    # style transfer
    _, sty_trg = g_model.encode(trg_img)
    img_fake, att_fake = g_model.decode(
        cont_src, torch.cat(sty_trg[0],dim=1))
    if use_attention:
        img_fake = img_fake*att_fake + src_img*(1-att_fake)
    outputs += [img_fake]
    if use_attention:
        outputs += [(att_fake.repeat(1,3,1,1) - 0.5)/0.5]

    # random sampling
    img_rnd, att_rnd = g_model.decode(
        cont_src, z_trg)
    if use_attention:
        img_rnd = img_rnd*att_rnd + src_img*(1-att_rnd)
    outputs += [img_rnd]

    outputs = torch.cat(outputs, dim=0)
    outputs = denormalize(outputs)
    vutils.save_image(outputs.cpu(), fname, nrow=bsz, padding=0)
    g_model.train()


# ----------- others ----------- #

def style_replace(gmm_src, gmm_trg, z_src, z_trg):
    """
    :gmm_src: Tensor, [N, S]
    :gmm_trg: Tensor, [N, S]
    :z_src: Tensor, [N, S, D]
    :z_trg: Tensor, [N, S, D]
    """
    mark = gmm_src==gmm_trg
    for i in range(gmm_src.size(0)):
        for j in range(gmm_src.size(1)):
            if mark[i,j]:
                z_trg[i, j] = z_src[i, j]
    return z_trg.detach()