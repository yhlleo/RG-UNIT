# --------------------------------------------------------
# RG-UNIT
# Written by Yahui Liu (yahui.liu@unitn.it)
# --------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

def criterion_ce(
    logits, # prediction
    target, # ground truth
    dataset='CelebA'
):
    """Compute binary or softmax cross entropy loss."""
    criterion = F.binary_cross_entropy_with_logits if dataset in ['CelebA'] else F.cross_entropy
    return criterion(logits, target)

def criterion_adv(
    x_real=None, # real image
    x_fake=None, # fake image
    gan_type='lsgan', # we support lsgan and wgan only
    is_D=True # True, training the discrimintaor; False, training the generator
):
    if gan_type == "lsgan":
        if is_D:
            return torch.mean(x_fake**2) + torch.mean((x_real-1)**2)
        else:
            return torch.mean((x_fake-1)**2)
    else:
        if is_D:
            return torch.mean(x_fake) - torch.mean(x_real)
        else:
            return -torch.mean(x_fake)

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def calc_d_loss(
    d_model,
    input_real,
    input_fake, 
    real_cls,
    lambda_adv=1.0, 
    lambda_cls=1.0,
    lambda_gp=0.0,
    gan_type='lsgan',
    dataset='CelebA',
    device=None
):
    loss = 0.0
    # calculate the loss to train D
    real_outs = d_model(input_real)
    fake_outs = d_model(input_fake)
    
    for routs, fouts in zip(real_outs, fake_outs):
        rsrc, rcls = routs
        fsrc, fcls = fouts
        loss += criterion_adv(x_real=rsrc, x_fake=fsrc, gan_type=gan_type, is_D=True) * lambda_adv
        loss += criterion_ce(rcls, real_cls, dataset) * lambda_cls

    if gan_type == "wgan":
        # Compute loss for gradient penalty.
        alpha = torch.rand(input_real.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * input_real.data + (1 - alpha) * input_fake.data).requires_grad_(True)
        gp_outs = d_model(x_hat)[0]
        loss += gradient_penalty(gp_outs[0], x_hat, device) * lambda_gp
    return loss

def calc_g_loss(
    d_model, 
    input_fake, 
    target_cls, 
    lambda_adv=1.0, 
    lambda_cls=1.0,
    gan_type='lsgan',
    dataset='CelebA'
):
    loss = 0.0
    # calculate the loss to train G
    fake_outs = d_model(input_fake)
    for fouts in fake_outs:
        fsrc, fcls = fouts
        loss += criterion_adv(x_fake=fsrc, gan_type=gan_type, is_D=False) * lambda_adv
        loss += criterion_ce(fcls, target_cls, dataset) * lambda_cls
    return loss

def criterion_earth_mover_distance(pred_mu, gt_mu):
    """
    :param: pred, extracted attribute vector with shape [N, d, V]
    :param: mus, mean tensor with shape [N, d, 1]
    """
    return torch.pow(pred_mu-gt_mu, 2).mean(dim=2).mean()

def criterion_kl_distance(pred_mu, pred_sigma, gt_mu, gt_sigma=None, device=None):
    """
    :param: pred_mu, extracted attribute vector with shape [N, d, V]
    :param: pred_sigma, extracted attribute vector with shape [N, d, V]
    :param: mus, mean tensor with shape [N, d, 1]
    """
    sigma = torch.tensor(0.25).to(device) if gt_sigma is None else gt_sigma
    return (0.5 * (torch.log(sigma/pred_sigma.exp()) + \
        (pred_sigma.exp() + torch.pow(pred_mu-gt_mu, 2))/sigma - 1.0)).sum(dim=1).mean()

def criterion_kl(
    pred_mu, 
    gt_mu,
    pred_sigma=None,
    gt_sigma=None,
    kl_mode="kl", # KL or Earth mover
    device=None
): 
    if kl_mode == 'kl':
        return criterion_kl_distance(pred_mu, pred_sigma, gt_mu, gt_sigma, device)
    else:
        return criterion_earth_mover_distance(pred_mu, gt_mu)


def criterion_l1(x, y):
    return torch.mean(torch.abs(x - y))

def vgg_preprocess(x, device):
    x   = (x + 1)/2.0 # [-1, 1] -> [0, 1]
    mu  = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
    x   = (x - mu) / std
    return x

def criterion_vgg(img1, img2, vgg_model, instancenorm, device):
    img1_vgg = vgg_preprocess(img1, device)
    img2_vgg = vgg_preprocess(img2, device)
    feat1    = vgg_model(img1_vgg)
    feat2    = vgg_model(img2_vgg)
    return torch.mean((instancenorm(feat1) - instancenorm(feat2)) ** 2)


def triplet_margin(margin=0.2, p=2):
    return nn.TripletMarginLoss(margin=margin, p=p)

class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine
