import torch
import torch.nn as nn
from .softsplat import ModuleSoftsplat


class SymmetricSplatting(nn.Module):

    def __init__(self):
        super(SymmetricSplatting, self).__init__()
        self.softsplat = ModuleSoftsplat("SymmetricSoftMax")

    def forward(self, 
                ftensor, 
                fflow,
                fmetric,
                btensor,
                bflow,
                bmetric,
                t,
                N
                ):
        alpha = t / (N * 1.0)
        alpha = alpha.reshape(-1, 1, 1, 1)
        ftensor_splatted = self.softsplat(ftensor, fflow, fmetric) * (1.0 - alpha)
        tensor_norm = ftensor_splatted[:,-1:, :, :]
        btensor_splatted = self.softsplat(btensor, bflow, bmetric) * alpha
        tensor_norm = tensor_norm + btensor_splatted[:,-1:, :, :] 
        tensor_norm[tensor_norm == 0.0] = 1.0
        return (ftensor_splatted[:, :-1, :, :] + btensor_splatted[:, :-1, :, :]) / tensor_norm
