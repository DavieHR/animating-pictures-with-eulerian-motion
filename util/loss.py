import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
from torch import autograd

from contextual_loss import ContextualLoss, ContextualBilateralLoss
from util.flow_utils import compute_meshgrid

class FeatureMatchLoss(nn.Module):
    """FeatureMatch.
    """
    def __init__(self):
        super(FeatureMatchLoss, self).__init__()

        self.loss = L1Loss()

    def forward(self, features_lhs, features_rhs):
        
        loss_total = torch.zeros((1)).to(features_lhs[0])
        for lhs, rhs in zip(features_lhs, features_rhs):
            loss_total += self.loss(lhs, rhs)

        return loss_total

class StyleGanLoss(nn.Module):
    """StyleGan v2 loss.
       copy the implement from pytorch code.
    """
    def __init__(self):
        super(StyleGanLoss, self).__init__()
        self.loss = FeatureMatchLoss()

    def forward(self, fake, real=None, feature_fake = None, feature_real = None):
        if real is not None:
            return  F.softplus(-real).mean() + F.softplus(fake).mean()
        else:
            return F.softplus(-fake).mean(), self.loss(feature_fake, feature_real)

class HingeGanLoss(nn.Module):
    """StyleGan v2 loss.
       copy the implement from pytorch code.
    """
    def __init__(self):
        super(HingeGanLoss, self).__init__()
        self.loss = FeatureMatchLoss()

    def forward(self, fake, real=None, feature_fake = None, feature_real = None):
        if real is not None:
            return nn.ReLU()(1.0 - real).mean() + nn.ReLU()(1.0 + fake).mean()
        else:
            return -fake.mean(), self.loss(feature_fake, feature_real)

class R1_loss(nn.Module):
    """regularize
    """
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

class flow_weakly_sample_loss(nn.Module):
    def __init__(self):
        super(flow_weakly_sample_loss, self).__init__()
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        #self.Warp = Warp()

    def forward(self, flow, flow_gt, img1, label, h_ori, w_ori):
        n,c,h,w = flow.shape
        #warped_img1 = self.Warp.forward(img1, flow)

        grid = compute_meshgrid(img1.shape).to(img1) + flow
        warped_image = F.grid_sample(img1, grid.permute(0,2,3,1))
        loss = self.criterion(label, warped_image)
        return loss

class flow_apply_loss(nn.Module):
    def __init__(self):
        super(flow_apply_loss, self).__init__()
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        #self.Warp = Warp()

    def forward(self, flow, flow_gt, img1, label, h_ori, w_ori):
        n,c,h,w = flow.shape
        #warped_img1 = self.Warp.forward(img1, flow)

        grid = compute_meshgrid(img1.shape).to(img1) + flow
        warped_image = F.grid_sample(img1, grid.permute(0,2,3,1))

        grid = compute_meshgrid(img1.shape).to(img1) + flow_gt
        warped_image_gt = F.grid_sample(img1, grid.permute(0,2,3,1))

        loss = self.criterion(warped_image_gt, warped_image)
        return loss

class flow_supervised_loss(nn.Module):

    def __init__(self):
        super(flow_supervised_loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, flow, flow_gt, img, label,h_ori, w_ori):
        n,c,h,w = flow.shape
        #flow_gt_norm = (flow_gt - flow_gt.contiguous().view(flow_gt.size()[:2] + (-1,)).mean(dim=-1).view(flow_gt.size()[:2] + (1,1,))) / (flow_gt.contiguous().view(flow_gt.size()[:2] + (-1,)).std(dim=-1).view(flow_gt.size()[:2] + (1,1,)) + 1e-4)
        #norm_factor = torch.cat((torch.ones((n,1,h,w)) * w_ori, torch.ones(n,1,h,w) * h_ori), dim = 1).to(flow)
        loss = self.criterion(flow, flow_gt)
        return loss

class flow_vgg_supervised_loss(nn.Module):

    def __init__(self):
        super(flow_supervised_loss, self).__init__()
        self.criterion = VggLoss()

    def forward(self, flow, flow_gt, img, label,h_ori, w_ori):
        n,c,h,w = flow.shape
        #norm_factor = torch.cat((torch.ones((n,1,h,w)) * w_ori, torch.ones(n,1,h,w) * h_ori), dim = 1).to(flow)
        loss = self.criterion(flow, flow_gt)
        return loss

class VggLoss(ContextualLoss):
    """ Vgg loss
    """
    def __init__(self,
                size: int = 512,
                vgg_layers: str = 'relu4_4',
                vgg_weights: list = [1]
                ):
        super(VggLoss, self).__init__(use_vgg = True, vgg_layer = vgg_layers)
        self.loss = MSELoss()
        self._size = size
        self.vgg_weights = vgg_weights

    def forward(self, x, y):
        if x.size(2) < 128:
            return torch.zeros((1)).to(x)
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5
        if x.size(2) > self._size:
            x = F.interpolate(x, (self._size,self._size))
            y = F.interpolate(y, (self._size, self._size))

        assert x.shape[1] == 3 and y.shape[1] == 3,\
            'VGG model takes 3 chennel images.'
        self.vgg_model = self.vgg_model.to(x)
        # normalization
        x = x.sub(self.vgg_mean.detach().to(x)).div(self.vgg_std.detach().to(x))
        y = y.sub(self.vgg_mean.detach().to(x)).div(self.vgg_std.detach().to(x))

        # picking up vgg feature maps
        if isinstance(self.vgg_layer,list):
            vgg_layers = self.vgg_layer
        else:
            vgg_layers = [self.vgg_layer]
        loss = torch.zeros((1)).to(x)
        f_x, f_y = self.vgg_model(x), self.vgg_model(y)
        for (weight,vgg_layer) in zip(self.vgg_weights, vgg_layers):
            x = getattr(f_x, vgg_layer)
            y = getattr(f_y, vgg_layer)
            loss += weight * self.loss(x,y)
        return loss


class flow_regularized_loss(nn.Module):
    def __init__(self, reg_value = 1.0):
        super(flow_regularized_loss, self).__init__()
        self.reg_value = reg_value
        self.loss = MSELoss()

    def forward(self, x, y, *args):
        reg_value_tensor = torch.ones_like(x) * self.reg_value
        return self.loss(x, reg_value_tensor)
