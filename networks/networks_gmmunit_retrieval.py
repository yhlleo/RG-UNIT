import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networks import FocalLoss

##################################################################################
# Discriminator
##################################################################################
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params, device=None):
        super(MsImageDis, self).__init__()
        self.n_layer    = params['n_layer']
        self.gan_type   = params['gan_type']
        self.dim        = params['dim']
        self.norm       = params['norm']
        self.activ      = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type   = params['pad_type']
        self.num_cls    = params['num_cls']
        self.input_dim  = input_dim
        self.image_size = params['image_size']
        self.dataset    = params['dataset']
        self.device     = device if device is not None else torch.device('cpu')
        self.cnns_feat  = nn.ModuleList()
        self.cnns_src   = nn.ModuleList()
        self.cnns_cls   = nn.ModuleList()
        for i in range(self.num_scales):
            net_feat, net_src, net_cls = self._make_net(self.image_size//(2**i))
            self.cnns_feat.append(net_feat)
            self.cnns_src.append(net_src)
            self.cnns_cls.append(net_cls)

        self.cross_entropy = nn.CrossEntropyLoss()

    def _classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset in ['CelebA']:
            return F.binary_cross_entropy_with_logits(logit, target, reduction='mean')
        else: # mode in ['edges2shoes']
            return self.cross_entropy(logit, target)

    def _make_net(self, im_size):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        pre_dim = dim
        for i in range(self.n_layer - 1):
            dim = min(dim*2, 512)
            cnn_x += [Conv2dBlock(pre_dim, dim, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            pre_dim = dim   
        cnn_x = nn.Sequential(*cnn_x)
        cnn_src = nn.Conv2d(dim, 1, 1, 1, 0)
        cnn_cls = nn.Conv2d(dim, self.num_cls, kernel_size=im_size//(2**self.n_layer), stride=1, padding=0, bias=False)
        return cnn_x, cnn_src, cnn_cls

    def forward(self, x):
        outputs = []
        for net_feat, net_src, net_cls in zip(self.cnns_feat, self.cnns_src, self.cnns_cls):
            out_feat = net_feat(x)
            out_src  = net_src(out_feat)
            out_cls  = net_cls(out_feat)
            out_cls  = out_cls.view(out_cls.size(0), out_cls.size(1))
            outputs.append([out_src, out_cls])
            #x = self.downsample(x)
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        return outputs

    def calc_dis_loss(self, input_fake, input_real, real_cls, weight_gan=1.0, weight_cls=1.0):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0.0

        for it, (out_fake, out_real) in enumerate(zip(outs0, outs1)):
            out_src_fake, out_cls_fake = out_fake
            out_src_real, out_cls_real = out_real

            # gan loss
            if self.gan_type == 'lsgan':
                loss += (torch.mean((out_src_fake - 0)**2) + torch.mean((out_src_real - 1)**2)) * weight_gan
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out_src_fake.data).to(self.device), requires_grad=False)
                all1 = Variable(torch.ones_like(out_src_real.data).to(self.device), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out_src_fake), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out_src_real), all1)) * weight_gan
            elif self.gan_type == 'wgan':
                loss += (torch.mean(out_src_fake) - torch.mean(out_src_real)) * weight_gan
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

            # classification loss
            loss += self._classification_loss(out_cls_real, real_cls, self.dataset)
        return loss

    def calc_gen_loss(self, input_fake, target_cls, weight_gan=1.0, weight_cls=1.0):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, out in enumerate(outs0):
            out_src_fake, out_cls_fake = out

            if self.gan_type == 'lsgan':
                loss += torch.mean((out_src_fake - 1)**2) * weight_gan # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out_src_fake.data).to(self.device), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out_src_fake), all1)) * weight_gan
            elif self.gan_type == 'wgan':
                loss +=  -torch.mean(out_src_fake) * weight_gan
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

            loss += self._classification_loss(out_cls_fake, target_cls, self.dataset)*weight_cls
        return loss

##################################################################################
# Generator
##################################################################################

class AdaINGenRaw(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGenRaw, self).__init__()
        dim           = params['dim']
        n_res         = params['n_res']
        activ         = params['activ']
        pad_type      = params['pad_type']
        mlp_dim       = params['mlp_dim']
        use_attention = params['use_attention']
        c_dim         = params['c_dim']
        num_cls       = params['num_cls']

        use_map       = params['use_map']
        use_dsn       = params['use_dsn']

        style_dim     = c_dim * num_cls
        style_downsample   = params['style_downsample']
        content_downsample = params['content_downsample']

        # style encoder
        self.enc_style = StyleEncoder(style_downsample, input_dim, dim, 
            norm='none', activ=activ, pad_type=pad_type, 
            c_dim=c_dim, num_class=num_cls, use_map=use_map, 
            use_dsn=use_dsn)

        # content encoder
        self.enc_content = ContentEncoder(content_downsample, n_res, 
            input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(content_downsample, n_res, self.enc_content.output_dim, 
            input_dim, res_norm='adain', activ=activ, pad_type=pad_type, 
            use_attention=use_attention)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, 
            norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, outputs = self.encode(images)
        mus = torch.cat(outputs[0], dim=1)
        images_recon = self.decode(content, mus)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        #mus, logvar = self.enc_style(images)
        outputs = self.enc_style(images)
        content = self.enc_content(images)
        return content, outputs

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim           = params['dim']
        n_res         = params['n_res']
        activ         = params['activ']
        pad_type      = params['pad_type']
        mlp_dim       = params['mlp_dim']
        use_attention = params['use_attention']
        c_dim         = params['c_dim']
        num_cls       = params['num_cls']

        use_map       = params['use_map']
        use_dsn       = params['use_dsn']

        style_dim     = c_dim * num_cls
        style_downsample   = params['style_downsample']
        content_downsample = params['content_downsample']
        num_ret       = params['num_results']

        # style encoder
        self.enc_style = StyleEncoder(content_downsample, input_dim, dim, 
            norm='none', activ=activ, pad_type=pad_type, 
            c_dim=c_dim, num_class=num_cls, use_map=use_map, 
            use_dsn=use_dsn)

        # content encoder
        self.enc_content = ContentEncoder(content_downsample, n_res, 
            input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(content_downsample, n_res, self.enc_content.output_dim, 
            input_dim, res_norm='adain', activ=activ, pad_type=pad_type, 
            use_attention=use_attention)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, 
            norm='none', activ=activ)

        self.retrieved_fuse = nn.Conv2d(
            self.enc_content.output_dim*num_ret, 
            self.enc_content.output_dim,
            kernel_size=3, padding=1, stride=1)

        self.conv = nn.Conv2d(
            self.enc_content.output_dim*2, 
            self.enc_content.output_dim, 
            kernel_size=3, padding=1, stride=1)

    def forward(self, images, retrieved_images):
        # reconstruct an image
        content, outputs = self.encode(images)
        feats_ret = self.encode_retrieved(retrieved_images)
        images_recon = self.decode(content, torch.cat(outputs[0],dim=1), feats_ret)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        outputs = self.enc_style(images)
        content = self.enc_content(images)
        return content, outputs

    def encode_retrieved(self, retrieved_images):
        feats = self.enc_content(retrieved_images)
        feats = feats.contiguous().view(1,-1,*feats.size()[2:])
        return self.retrieved_fuse(feats)
    
    def decode(self, content, style, feats_ret):
        # decode content and style codes to an image
        adain_params = self.mlp(style) 
        self.assign_adain_params(adain_params, self.dec)
        feats = self.conv(torch.cat([content, feats_ret],dim=1))
        images = self.dec(feats)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def freezen_params(self):
        self.enc_style.eval()
        for param in self.enc_style.parameters():
            param.requires_grad = False
        self.enc_content.eval()
        for param in self.enc_content.parameters():
            param.requires_grad = False
        self.mlp.eval()
        for param in self.mlp.parameters():
            param.requires_grad = False

##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, norm, activ, pad_type, c_dim, num_class, 
        use_map=False, use_dsn=False):
        super(StyleEncoder, self).__init__()
        self.num_class = num_class
        self.use_dsn = use_dsn
        self.use_map = use_map

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *=2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model = nn.Sequential(*self.model)

        if self.use_map:
            self.mapping = nn.Sequential(nn.Linear(dim, dim),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(p=0.1),
                                         nn.Linear(dim, dim),
                                         nn.ReLU(inplace=True))

        self.fcs = nn.ModuleList()
        self.fcvars = nn.ModuleList()
        for i in range(self.num_class):
            self.fcs.append(nn.Linear(dim, c_dim))
            self.fcvars.append(nn.Linear(dim, c_dim))
        self.output_dim = dim

        # Using deeply-supervised nets
        if self.use_dsn:
            self.fcs_dsn = nn.ModuleList()
            self.fcvars_dsn = nn.ModuleList()
            for i in range(self.num_class):
                self.fcs_dsn.append(nn.Linear(dim, c_dim))
                self.fcvars_dsn.append(nn.Linear(dim, c_dim))

    def forward(self, x):
        feats  = self.model(x)
        feats_ = feats.view(x.size(0), -1)
        if self.use_dsn:
            fcs_dsn, fcvars_dsn = [], []
            for i in range(self.num_class):
                fcs_dsn.append(self.fcs_dsn[i](feats_))
                fcvars_dsn.append(self.fcvars_dsn[i](feats_))

        if self.use_map:
            feats_ = self.mapping(feats_)
        fcs, fcvars = [], []
        for i in range(self.num_class):
            fcs.append(self.fcs[i](feats_))
            fcvars.append(self.fcvars[i](feats_))
        outputs = [fcs, fcvars]

        if self.use_dsn:
            outputs.extend([fcs_dsn, fcvars_dsn])
        return outputs


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', 
        activ='relu', pad_type='zero', use_attention=False):
        super(Decoder, self).__init__()
        self.use_attention = use_attention
        model = []
        # AdaIN residual blocks
        model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            model += [nn.Upsample(scale_factor=2, mode='nearest'),
                      Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model = nn.Sequential(*model)
        self.image_content   = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        
        image_attention = []
        image_attention += [Conv2dBlock(dim, 1, 7, 1, 3, norm='none', activation='sigmoid', pad_type=pad_type)]
        self.image_attention = nn.Sequential(*image_attention)

    def forward(self, x):
        feats     = self.model(x)
        content   = self.image_content(feats)
        attention = None

        if self.use_attention:
            attention = self.image_attention(feats)
        return content, attention

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=False)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)