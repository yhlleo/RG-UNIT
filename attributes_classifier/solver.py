import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable

from resnets import resnet50
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_scheduler

class Solver(nn.Module):
    def __init__(self, configs, device=None):
        super(Solver, self).__init__()
        lr = configs['lr']
        momentum = configs['momentum']
        weight_decay = configs['weight_decay']

        self.device = device if device is not None else torch.device('cpu')
        self.num_classes = configs['num_classes']

        # Initiate the network
        self.model = resnet50(
            pretrained=configs['use_pretrained'], 
            num_classes=configs['num_classes'])

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Setup the optimizer
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, 
            momentum=momentum, weight_decay=weight_decay)
        self.scheduler = get_scheduler(self.opt, configs)

    def forward(self, x, label):
        self.opt.zero_grad()
        preds = self.model(x)
        self.loss = self.criterion(preds, label)
        self.loss.backward()
        self.opt.step()

    def test(self, x, label):
        self.eval()
        with torch.no_grad():
            preds = self.model(x)
        self.train()
        return self.correct_per_class(preds, label)

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def resume(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict['class'])
        print('Checkpoint loaded')

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        path = os.path.join(snapshot_dir, 'class_%08d.pt' % (iterations + 1))
        torch.save({'class': self.model.state_dict()}, path)

    def save_best(self, snapshot_dir):
        path = os.path.join(snapshot_dir, 'best_model.pt')
        torch.save({'class': self.model.state_dict()}, path)

    def mean_acc(self, image, label):
        self.eval()
        out = self.model(image)
        out = torch.sigmoid(out)
        out[out>=0.5] = 1.0
        out[out<0.5]  = 0.0
        self.train()
        return torch.mean((out==label).float())

    def correct_per_class(self, out, label):
        out = torch.sigmoid(out)
        out[out>=0.5] = 1.0
        out[out<0.5]  = 0.0
        return (out==label).float()

