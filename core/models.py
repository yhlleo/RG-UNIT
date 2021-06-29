# --------------------------------------------------------
# RG-UNIT
# Written by Yahui Liu (yahui.liu@unitn.it)
# --------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

from .basic_modules import (
    Conv2dBlock,
    ResBlock,
    ResBlocks,
    LinearBlock,
    MLP,
    AdaptiveInstanceNorm2d,
    LayerNorm,
    SpectralNorm,
    StyleEncoder,
    ContentEncoder,
    Decoder
)

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, args):
        super(AdaINGen, self).__init__()
        input_dim     = args['input_dim']
        gf_dim        = args['gf_dim']
        n_res         = args['n_res']
        activ         = args['activ']
        pad_type      = args['pad_type']
        mlp_dim       = args['mlp_dim']
        attr_dim      = args['attr_dim']
        num_cls       = args['num_cls']
        n_downsample  = args['n_downsample']
        use_attention = args['use_attention']
        sty_dim       = attr_dim * num_cls

        # style encoder
        self.enc_style = StyleEncoder(
            input_dim=input_dim, 
            dim=gf_dim, 
            n_downsample=4,
            norm='none', 
            activ=activ, 
            pad_type=pad_type, 
            attr_dim=attr_dim, 
            num_class=num_cls
        )

        # content encoder
        self.enc_content = ContentEncoder(
            input_dim=input_dim, 
            dim=gf_dim,
            n_downsample=n_downsample, 
            n_res=n_res, 
            norm='in', 
            activ=activ, 
            pad_type=pad_type
        )

        # decoder
        self.dec = Decoder(
            dim=self.enc_content.output_dim, 
            output_dim=input_dim, 
            n_upsample=n_downsample, 
            n_res=n_res, 
            res_norm='adain', 
            activ=activ, 
            pad_type='zero', 
            use_attention=use_attention
        )

        # MLP to generate AdaIN parameters
        self.mlp = MLP(
            input_dim=sty_dim, 
            output_dim=self.get_num_adain_params(self.dec), 
            dim=mlp_dim, 
            n_blk=3, 
            norm='none', 
            activ='relu'
        )

    def forward(self, x):
        # reconstruct an image
        content, styles = self.encode(x)
        return self.decode(content, torch.cat(styles[0],dim=1))

    def encode(self, x):
        # encode an image to its content and style codes
        styles  = self.enc_style(x)
        content = self.enc_content(x)
        return content, styles

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        return self.dec(content)

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


class AdaINGenRet(nn.Module):
    # AdaIN auto-encoder architecture + retrieval system
    def __init__(self, args):
        super(AdaINGen, self).__init__()
        input_dim     = args['input_dim']
        gf_dim        = args['gf_dim']
        n_res         = args['n_res']
        activ         = args['activ']
        pad_type      = args['pad_type']
        mlp_dim       = args['mlp_dim']
        attr_dim      = args['attr_dim']
        num_cls       = args['num_cls']
        n_downsample  = args['n_downsample']
        use_attention = args['use_attention']
        num_ret       = args['num_results']
        sty_dim       = attr_dim * num_cls

        # style encoder
        self.enc_style = StyleEncoder(
            input_dim=input_dim, 
            dim=gf_dim, 
            n_downsample=4,
            norm='none', 
            activ=activ, 
            pad_type=pad_type, 
            attr_dim=attr_dim, 
            num_class=num_cls
        )

        # content encoder
        self.enc_content = ContentEncoder(
            input_dim=input_dim, 
            dim=gf_dim,
            n_downsample=n_downsample, 
            n_res=n_res, 
            norm='in', 
            activ=activ, 
            pad_type=pad_type
        )

        # decoder
        self.dec = Decoder(
            dim=self.enc_content.output_dim, 
            output_dim=input_dim, 
            n_upsample=n_downsample, 
            n_res=n_res, 
            res_norm='adain', 
            activ=activ, 
            pad_type='zero', 
            use_attention=use_attention
        )

        # MLP to generate AdaIN parameters
        self.mlp = MLP(
            input_dim=sty_dim, 
            output_dim=self.get_num_adain_params(self.dec), 
            dim=mlp_dim, 
            n_blk=3, 
            norm='none', 
            activ='relu'
        )

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
        content, styles = self.encode(images)
        feats_ret = self.encode_retrieved(retrieved_images)
        return self.decode(content, torch.cat(styles[0],dim=1), feats_ret)

    def encode(self, images):
        # encode an image to its content and style codes
        styles  = self.enc_style(images)
        content = self.enc_content(images)
        return content, styles

    def encode_retrieved(self, retrieved_images):
        sz = retrieved_images.size() # [bsz, k, 3, H, W]
        feats = self.enc_content(retrieved_images.view(-1, *sz[2:])) # [bsz*k, C, H/K, W/K]
        feats = feats.contiguous().view(*sz[:2], feats.size()[1:]) # [bsz, k, C, H/K, W/K]
        feats = feats.view(sz[0], -1, feats.size()[3:]) # [bsz, k*C, H/K, W/K]
        return self.retrieved_fuse(feats)
    
    def decode(self, content, style, retrieved_images):
        # extract features from retrieved images
        feats_ret = self.encode_retrieved(retrieved_images)
        # decode content and style codes to an image
        adain_params = self.mlp(style) 
        self.assign_adain_params(adain_params, self.dec)
        feats = self.conv(torch.cat([content, feats_ret], dim=1))
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


##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, args):
        super(MsImageDis, self).__init__()
        self.n_layer   = args['n_layer']
        self.df_dim    = args['df_dim']
        self.norm      = args['norm']
        self.activ     = args['activ']
        self.pad_type  = args['pad_type']
        self.num_cls   = args['num_cls']
        self.input_dim = args['input_dim']
        num_scales     = args['num_scales']
        img_size     = args['img_size']

        self.cnns_feat  = nn.ModuleList()
        self.cnns_src   = nn.ModuleList()
        self.cnns_cls   = nn.ModuleList()
        for i in range(num_scales):
            net_feat, net_src, net_cls = self._make_net(img_size//(2**i))
            self.cnns_feat.append(net_feat)
            self.cnns_src.append(net_src)
            self.cnns_cls.append(net_cls)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def _make_net(self, img_size):
        dim = self.df_dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        pre_dim = dim
        for i in range(self.n_layer - 1):
            dim = min(dim*2, 512)
            cnn_x += [Conv2dBlock(pre_dim, dim, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            pre_dim = dim   
        cnn_x = nn.Sequential(*cnn_x)
        cnn_src = nn.Conv2d(dim, 1, 1, 1, 0)
        cnn_cls = nn.Conv2d(dim, self.num_cls, kernel_size=img_size//(2**self.n_layer), stride=1, padding=0, bias=False)
        return cnn_x, cnn_src, cnn_cls

    def forward(self, x):
        outputs = []
        for net_feat, net_src, net_cls in zip(self.cnns_feat, self.cnns_src, self.cnns_cls):
            out_feat = net_feat(x)
            out_src  = net_src(out_feat)
            out_cls  = net_cls(out_feat)
            out_cls  = out_cls.view(out_cls.size(0), out_cls.size(1))
            outputs.append([out_src, out_cls])
            x = self.downsample(x)
            #x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        return outputs


class RetrievalNet(nn.Module):
    def __init__(self, args):
        super(RetrievalNet, self).__init__()
        embed_dim   = args['embed_dim']
        hidden_size = args['hidden_size']
        num_layers  = args['num_layers']
        activ       = args['activ']
        norm        = args['norm']
        pad_type    = args['pad_type']
        style_dim   = args['attr_dim'] * args['num_cls']

        # Convs for the content feature map
        # Content feature map is [256x32x32]
        conv_down = [Conv2dBlock(256, 128, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type), # [N, 128, 32, 32]
                     Conv2dBlock(128, 128, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type), # [N, 128, 16, 16]
                     Conv2dBlock(128, 64, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type),  # [N, 64, 16, 16]
                     Conv2dBlock(64, 64, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),   # [N, 64, 8, 8]
                     Conv2dBlock(64, 32, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type),   # [N, 32, 8, 8]
                     Conv2dBlock(32, 32, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]   # [N, 32, 4, 4]
        self.conv_down = nn.Sequential(*conv_down)

        # MLP
        self.model = []
        self.model += [LinearBlock(512+style_dim, hidden_size, norm='none', activation=activ)] # flatten [32x4x4] feature map + the attributes encoding
        for i in range(num_layers - 2):
            self.model += [LinearBlock(hidden_size, hidden_size, norm='none', activation=activ),
                           nn.Dropout(p=0.1)]
        self.model += [LinearBlock(hidden_size, embed_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x_cont, x_sty):
        x_cont = self.conv_down(x_cont)
        # Flatten the 32x4x4 tensor
        x_cont = x_cont.contiguous().view(x_cont.size(0), -1)
        x = torch.cat((x_cont, x_sty), dim=1)
        out = self.model(x)
        return out

    def freezen_params(self):
        self.conv_down.eval()
        for param in self.conv_down.parameters():
            param.requires_grad = False
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

