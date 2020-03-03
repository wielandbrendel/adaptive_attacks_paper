import torch
import torch.nn as nn
import numpy as np

me_channel = 'concat'
svd = True
maskp = 0.5
svdprob = 0.8
mu = 1

mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

class usvt(torch.autograd.Function):
    """ME-Net layer with universal singular value thresholding (USVT) approach.
    The ME preprocessing is embedded into a Function subclass for adversarial training.
    ----------
    Chatterjee, S. et al. Matrix estimation by universal singular value thresholding. 2015.
    https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input):
        device = input.device
        batch_num, c, h, w = input.size()
        
        output = torch.zeros_like(input).cpu().numpy()
        
        for i in range(batch_num):
            img = (input[i] * 2 - 1).cpu().numpy()

            if me_channel == 'concat':
                img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
                mask = np.random.binomial(1, maskp, h * w * c).reshape(h, w * c)
                p_obs = len(mask[mask == 1]) / (h * w * c)
                
                if svd:
                    u, sigma, v = np.linalg.svd(img * mask)
                    S = np.zeros((h, w))
                    for j in range(int(svdprob * h)):
                        S[j][j] = sigma[j]
                    S = np.concatenate((S, np.zeros((h, w * 2))), axis=1)
                    W = np.dot(np.dot(u, S), v) / p_obs
                    W[W < -1] = -1
                    W[W > 1] = 1
                    est_matrix = (W + 1) / 2
                    for channel in range(c):
                        output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
                else:
                    est_matrix = ((img * mask) + 1) / 2
                    for channel in range(c):
                        output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
                
            else:
                mask = np.random.binomial(1, maskp, h * w).reshape(h, w)
                p_obs = len(mask[mask == 1]) / (h * w)
                for channel in range(c):
                    u, sigma, v = np.linalg.svd(img[channel] * mask)
                    S = np.zeros((h, w))
                    for j in range(int(svdprob * h)):
                        S[j][j] = sigma[j]
                    W = np.dot(np.dot(u, S), v) / p_obs
                    W[W < -1] = -1
                    W[W > 1] = 1
                    output[i, channel] = (W + 1) / 2

        output = output - mean
        output /= std
        output = torch.from_numpy(output).float().to(device)
        return output
        

    @staticmethod
    def backward(ctx, grad_output):
        # BPDA, approximate gradients
        return grad_output
    

class MENet(nn.Module):
    """ME-Net layer.
    To attack a trained ME-Net model, first load the checkpoint, then wrap the loaded model with ME layer.
    Example:
        model = checkpoint['model']
        menet_model = MENet(model)
    ----------
    https://pytorch.org/docs/stable/notes/extending.html
    """
    def __init__(self, model):
        super(MENet, self).__init__()
        self.model = model

    def forward(self, input):
        x = globals()["usvt"].apply(input)
        return self.model(x)
    
