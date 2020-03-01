import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()
        return hook
    
    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class Sparsify1D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify1D, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        k = int(self.sr*x.shape[1])
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1,0)
        comp = (x>=topval).to(x)
        return comp*x

class Sparsify1D_kactive(SparsifyBase):
    def __init__(self, k=1):
        super(Sparsify1D_kactive, self).__init__()
        self.k = k
    
    def forward(self, x):
        k = self.k
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1,0)
        comp = (x>=topval).to(x)
        return comp*x 


class Sparsify2D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D, self).__init__()
        self.sr = sparse_ratio

        self.preact = None
        self.act = None

    def forward(self, x):
        layer_size = x.shape[2]*x.shape[3]
        k = int(self.sr*layer_size)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:,:,-1]
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2,3,0,1)
        comp = (x>=topval).to(x)
        return comp*x


class Sparsify2D_vol(SparsifyBase):
    '''cross channel sparsify'''
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_vol, self).__init__()
        self.sr = sparse_ratio


    def forward(self, x):
        size = x.shape[1]*x.shape[2]*x.shape[3]
        k = int(self.sr*size)

        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:,-1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1,0).view_as(x)
        comp = (x>=topval).to(x)
        return comp*x

class Sparsify2D_kactive(SparsifyBase):
    '''cross channel sparsify'''
    def __init__(self, k):
        super(Sparsify2D_vol, self).__init__()
        self.k = k


    def forward(self, x):
        k = self.k
        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:,-1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1,0).view_as(x)
        comp = (x>=topval).to(x)
        return comp*x

class Sparsify2D_abs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_abs, self).__init__()
        self.sr = sparse_ratio


    def forward(self, x):
        layer_size = x.shape[2]*x.shape[3]
        k = int(self.sr*layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:,:,-1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2,3,0,1)
        comp = (absx>=topval).to(x)
        return comp*x

class Sparsify2D_invabs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_invabs, self).__init__()
        self.sr = sparse_ratio


    def forward(self, x):
        layer_size = x.shape[2]*x.shape[3]
        k = int(self.sr*layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2, largest=False)[0][:,:,-1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2,3,0,1)
        comp = (absx>=topval).to(x)
        return comp*x


class breakReLU(nn.Module):
    def __init__(self, sparse_ratio=5):
        super(breakReLU, self).__init__()
        self.h = sparse_ratio
        self.thre = nn.Threshold(0, -self.h)

    def forward(self, x):
        return self.thre(x)

class SmallCNN(nn.Module):
    def __init__(self, fc_in=3136, n_classes=10):
        super(SmallCNN, self).__init__()

        self.module_list = nn.ModuleList([nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                            nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                            Flatten(),
                            nn.Linear(fc_in, 100), nn.ReLU(),
                            nn.Linear(100, n_classes)])

    def forward(self, x):
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
        return x

    def forward_to(self, x, layer_i):
        for i in range(layer_i):
            x = self.module_list[i](x)
        return x


sparse_func_dict = {
    'reg':Sparsify2D,  #top-k value
    'abs':Sparsify2D_abs,  #top-k absolute value
    'invabs':Sparsify2D_invabs, #top-k minimal absolute value
    'vol':Sparsify2D_vol,  #cross channel top-k
    'brelu':breakReLU, #break relu
    'kact':Sparsify2D_kactive,
    'relu':nn.ReLU
}