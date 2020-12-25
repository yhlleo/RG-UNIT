import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .networks import LinearBlock, Conv2dBlock


class RetrievalNet(nn.Module):
    def __init__(self, params):
        super(RetrievalNet, self).__init__()

        embed_dim   = params['embed_dim']
        hidden_size = params['hidden_size']
        num_layers  = params['num_layers']
        activ       = params['activ']
        norm        = params['norm']
        pad_type    = params['pad_type']
        style_dim   = params['c_dim'] * params['num_cls']

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

    def forward(self, x_cont, x_att):
        x_cont = self.conv_down(x_cont)
        # Flatten the 32x4x4 tensor
        x_cont = x_cont.contiguous().view(x_cont.size(0), -1)
        x = torch.cat((x_cont, x_att), dim=1)
        out = self.model(x)
        return out

    def freezen_params(self):
        self.conv_down.eval()
        for param in self.conv_down.parameters():
            param.requires_grad = False
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

