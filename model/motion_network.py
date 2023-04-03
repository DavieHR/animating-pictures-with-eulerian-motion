import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

class Warp(nn.Module):
    def __init__(self):
        super(Warp, self).__init__()
        #self.softsplat = ModuleSoftsplat("average")

    def forward(self, img, flo, is_norm=False):
        # softsplat: first image, flow
        # warped_img = self.softsplat(img, flo, None)

        shape = img.shape
        grid = self.compute_meshgrid(shape, is_norm=is_norm).to(img)
        grid = grid + flo

        if not is_norm:
            N, C, H, W = shape
            grid[:, 0, :, :] = grid[:, 0, :, :] / max(W-1, 1)
            grid[:, 1, :, :] = grid[:, 1, :, :] / max(H-1, 1)

        # scale grid to [-1,1]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] - 1.0

        warped_img = self.get_warped_img(img, grid)

        return warped_img

    def compute_meshgrid(self, shape, is_norm=False):
        N, C, H, W = shape
        rows = torch.arange(0, H, dtype=torch.float32)
        cols = torch.arange(0, W, dtype=torch.float32)

        if is_norm:
            rows /= H-1
            cols /= W-1

        grid = torch.meshgrid(rows, cols)
        grid = torch.stack(grid[::-1]).unsqueeze(0)
        grid = torch.cat([grid for _ in range(N)], dim=0)

        return grid

    def get_warped_img(self, img, grid):
        return F.grid_sample(img, grid.permute((0, 2, 3, 1)), align_corners=True)

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    # 偏移量的大小当然就是通过光流数组中的数值大小体现出来的，而偏移的方向是通过光流数组中的正负体现出来的。
    # 在x方向上，正值表示物体向左移动，而负值表示物体向右移动；
    # 在y方向上，正值表示物体向上移动，而负值表示物体向下移动

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    grid = grid + flo

    # scale grid to [-1,1]
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(W-1, 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(H-1, 1) - 1.0

    grid = grid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, grid, align_corners=True)
    # mask = torch.ones_like(x)
    # mask = nn.functional.grid_sample(mask, grid, align_corners=True)

    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1

    return output  # * mask

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_deepfillv2(net, init_type='kaiming', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)

def regularizer_clip(net, c_min = -0.5, c_max = 0.5, eps = 1e-3):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            w = m.weight.data.clone()
            w[w > c_max] -= eps
            w[w < c_min] += eps
            m.weight.data = w

            if m.bias is not None:
                b = m.bias.data.clone()
                b[b > c_max] -= eps
                b[b < c_min] += eps
                m.bias.data = b

    net.apply(init_func)

class norm_line(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight

def get_norm_layer(norm_type='instance', weight=0.2):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'line':
        norm_layer = norm_line(weight)
    elif norm_type == 'None':
        norm_layer = nn.Identity
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', adaptive_global=True, gpu_ids=[], use_activation='tanh'):
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':
        netG = GlobalGeneratorPaper(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, use_activation=use_activation)
        # netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, use_activation=use_activation)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer, use_activation=use_activation)
    elif netG == 'unet':
        netG = UNet(3, 2, bilinear=True, use_activation=use_activation)
    elif netG == 'unet5':
        netG = UNet5(3, 2, bilinear=True, use_activation=use_activation)
    elif netG == 'unet6':
        netG = UNet6(3, 2, bilinear=True, use_activation=use_activation)
    elif netG == 'unet7':
        netG = UNet7(3, 2, bilinear=True, use_activation=use_activation)
    elif netG == 'flownet2sd':
        netG = FlowNet2SD(64, norm_layer)
    else:
        raise('generator not implemented!')
    # print(netG)
    assert(torch.cuda.is_available())
    netG.cuda(0)
    # netG.apply(weights_init)
    weights_init_deepfillv2(netG, init_type='normal', init_gain=0.02)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    #if len(gpu_ids) > 0:
    #    assert(torch.cuda.is_available())
    #    netD.cuda(gpu_ids[0])
    netD.cuda(0)
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps).mean()

def get_criterion(type):
    if type == 'MSELoss':
        criterion = nn.MSELoss()
    elif type == 'L1Loss':
        criterion = nn.L1Loss()
    elif type == 'CharbonnierLoss':
        criterion = charbonnier_loss
    elif type == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    else:
        print('No existing loss type!')
        exit(0)

    return criterion

class WarpLoss(nn.Module):
    def __init__(self, type='MSELoss'):
        super(WarpLoss, self).__init__()
        self.criterion = get_criterion(type)
        self.Warp = Warp()

    def forward(self, flow, img1, img2, is_norm=False):
        # warped_img1 = warp(img2, flow)
        warped_img1 = self.Warp.forward(img2, flow, is_norm=is_norm)
        # img1 = self.Warp.forward(img1, torch.zeros_like(flow), is_norm=is_norm)

        loss = self.criterion(img1, warped_img1)

        return loss, warped_img1

class CycleWarpLoss(nn.Module):
    def __init__(self, type='MSELoss'):
        super(CycleWarpLoss, self).__init__()
        self.criterion = get_criterion(type)
        self.Warp = Warp()

    def forward(self, flow, img, is_norm=False):
        # forewarped_img = warp(img, flow)
        # backwarped_img = warp(forewarped_img, -1*flow)
        forewarped_img = self.Warp.forward(img, flow, is_norm=is_norm)
        backwarped_img = self.Warp.forward(forewarped_img, -1*flow, is_norm=is_norm)

        loss = self.criterion(img, backwarped_img)

        return loss

class WarpGtLoss(nn.Module):
    def __init__(self, type='MSELoss'):
        super(WarpGtLoss, self).__init__()
        self.criterion = get_criterion(type)
        self.Warp = Warp()

    def forward(self, flow, flow_gt, img1, is_norm=False):
        warped_image = self.Warp.forward(img1, flow, is_norm=is_norm)
        warped_image_gt = self.Warp.forward(img1, flow_gt, is_norm=is_norm)

        loss = self.criterion(warped_image_gt, warped_image)

        return loss

class PixelLoss(nn.Module):
    def __init__(self, type='L1Loss'):
        super(PixelLoss, self).__init__()
        self.criterion = get_criterion(type)

    def forward(self, img, label):
        loss = self.criterion(img, label)

        return loss

class TVLoss(nn.Module):
    def __init__(self, type='MSELoss'):
        super(TVLoss, self).__init__()
        self.criterion = get_criterion(type)

    def forward(self, x):
        h_x, w_x = x.size()[2:]

        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x-1, :])
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x-1])
        # loss = torch.sum(h_tv) + torch.sum(w_tv)
        # return loss / h_tv.numel()

        h_tv = self.criterion(x[:, :, 1:, :], x[:, :, :h_x-1, :])
        w_tv = self.criterion(x[:, :, :, 1:], x[:, :, :, :w_x-1])
        loss = h_tv + w_tv
        return loss

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = torch.tensor(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = torch.tensor(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, flow, img1, img2):
        warped_img1 = warp(img2, flow)

        x_vgg, y_vgg = self.vgg(img1), self.vgg(warped_img1)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect', use_activation='tanh'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            # model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
            model_upsample += [TransposeConv2dLayer(ngf_global * 2, ngf_global, kernel_size=3, padding=1, scale_factor=2),
                               norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:                
                if use_activation == 'tanh':
                    model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
                elif use_activation == 'none':
                    model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]  # , nn.Tanh()]
                else:
                    print('error activation type!!!')
                    exit()

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            # get_norm_layer('instance')(mid_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # get_norm_layer('instance')(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_activation='tanh'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if use_activation[:4] == 'tanh':
            try:
                weight = float(use_activation.split('_')[-1])
            except:
                weight = 1.0
            model = [get_norm_layer('line', weight), nn.Tanh()]
            self.tanh = nn.Sequential(*model)
        elif use_activation == 'none':
            self.tanh = get_norm_layer('line', 1)
        else:
            print('error activation type!!!')
            exit()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 64
        x4 = self.down3(x3)   # 32
        x5 = self.down4(x4)   # 16
        x = self.up1(x5, x4)  # 32
        x = self.up2(x, x3)   # 64
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 256
        logits = self.outc(x)

        logits = self.tanh(logits)

        return logits

class UNet5(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_activation='tanh'):
        super(UNet5, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048 // factor)
        self.up0 = Up(2048, 1024 // factor, bilinear)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if use_activation[:4] == 'tanh':
            try:
                weight = float(use_activation.split('_')[-1])
            except:
                weight = 1.0
            model = [get_norm_layer('line', weight), nn.Tanh()]
            self.tanh = nn.Sequential(*model)
        elif use_activation == 'none':
            self.tanh = get_norm_layer('line', 1)
        else:
            print('error activation type!!!')
            exit()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 64
        x4 = self.down3(x3)   # 32
        x5 = self.down4(x4)   # 16
        x6 = self.down5(x5)   # 8
        x = self.up0(x6, x5)  # 16
        x = self.up1(x, x4)   # 32
        x = self.up2(x, x3)   # 64
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 256
        logits = self.outc(x)

        logits = self.tanh(logits)

        return logits

class UNet6(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_activation='tanh'):
        super(UNet6, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048 // factor)
        self.down6 = Down(1024, 2048 // factor)
        self.up00 = Up(2048, 2048 // factor, bilinear)
        self.up0 = Up(2048, 1024 // factor, bilinear)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if use_activation[:4] == 'tanh':
            try:
                weight = float(use_activation.split('_')[-1])
            except:
                weight = 1.0
            model = [get_norm_layer('line', weight), nn.Tanh()]
            self.tanh = nn.Sequential(*model)
        elif use_activation == 'none':
            self.tanh = get_norm_layer('line', 1)
        else:
            print('error activation type!!!')
            exit()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 64
        x4 = self.down3(x3)   # 32
        x5 = self.down4(x4)   # 16
        x6 = self.down5(x5)   # 8
        x7 = self.down6(x6)   # 4
        x = self.up00(x7, x6) # 8
        x = self.up0(x, x5)   # 16
        x = self.up1(x, x4)   # 32
        x = self.up2(x, x3)   # 64
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 256
        logits = self.outc(x)

        logits = self.tanh(logits)

        return logits

class UNet7(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_activation='tanh'):
        super(UNet7, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048 // factor)
        self.down6 = Down(1024, 2048 // factor)
        self.down7 = Down(1024, 2048 // factor)
        self.up000 = Up(2048, 2048 // factor, bilinear)
        self.up00 = Up(2048, 2048 // factor, bilinear)
        self.up0 = Up(2048, 1024 // factor, bilinear)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if use_activation[:4] == 'tanh':
            try:
                weight = float(use_activation.split('_')[-1])
            except:
                weight = 1.0
            model = [get_norm_layer('line', weight), nn.Tanh()]
            self.tanh = nn.Sequential(*model)
        elif use_activation == 'none':
            self.tanh = get_norm_layer('line', 1)
        else:
            print('error activation type!!!')
            exit()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 64
        x4 = self.down3(x3)   # 32
        x5 = self.down4(x4)   # 16
        x6 = self.down5(x5)   # 8
        x7 = self.down6(x6)   # 4
        x8 = self.down7(x7)   # 2
        x = self.up000(x8, x7)# 4
        x = self.up00(x, x6)  # 8
        x = self.up0(x, x5)   # 16
        x = self.up1(x, x4)   # 32
        x = self.up2(x, x3)   # 64
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 256
        logits = self.outc(x)

        logits = self.tanh(logits)

        return logits
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def conv(in_planes, out_planes, norm_layer=nn.BatchNorm2d, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
        norm_layer(out_planes),
        nn.LeakyReLU(0.2, inplace=False)
    )

def deconv(in_planes, out_planes, norm_layer):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        # nn.Upsample(scale_factor=2, mode='bilinear'),
        # conv(in_planes, out_planes),
        # norm_layer(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )

def predict_flow(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Tanh(),
    )

def i_conv(in_planes, out_planes, norm_layer, kernel_size=3, stride=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias),
        # norm_layer(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )

class FlowNet2SD(nn.Module):
    def __init__(self, ngf, norm):
        super(FlowNet2SD, self).__init__()
        self.batchNorm = norm
        self.conv0 = conv(3, ngf * 1, self.batchNorm)
        self.conv1 = conv(ngf * 1, ngf * 1, self.batchNorm, stride=2)  #128
        self.conv1_1 = conv(ngf * 1, ngf * 2, self.batchNorm)
        self.conv2 = conv(ngf * 2, ngf * 2, self.batchNorm, stride=2)  #64
        self.conv2_1 = conv(ngf * 2, ngf * 2, self.batchNorm)
        self.conv3 = conv(ngf * 2, ngf * 4, self.batchNorm, stride=2)  #32
        self.conv3_1 = conv(ngf * 4, ngf * 4, self.batchNorm)
        self.conv4 = conv(ngf * 4, ngf * 8, self.batchNorm, stride=2)  #16
        self.conv4_1 = conv(ngf * 8, ngf * 8, self.batchNorm)
        self.conv5 = conv(ngf * 8, ngf * 8, self.batchNorm, stride=2)  #8
        self.conv5_1 = conv(ngf * 8, ngf * 8, self.batchNorm)
        self.conv6 = conv(ngf * 8, ngf * 16, self.batchNorm, stride=2)  #4
        self.conv6_1 = conv(ngf * 16, ngf * 16, self.batchNorm)

        #>>>>>>>>>>>
        # self.conv7   = conv(ngf*16, ngf*16, self.batchNorm, stride=2)  #2
        # self.conv7_1 = conv(ngf*16, ngf*16, self.batchNorm)

        # self.deconv6 = deconv(ngf * 16, ngf * 16, self.batchNorm)
        #<<<<<<<<<<<

        self.deconv5 = deconv(ngf * 16, ngf * 8, self.batchNorm)
        self.deconv4 = deconv(ngf * 16 + 2, ngf * 4, self.batchNorm)
        self.deconv3 = deconv(ngf * 8 + ngf * 4 + 2, ngf * 2, self.batchNorm)
        self.deconv2 = deconv(ngf * 4 + ngf * 2 + 2, ngf * 1, self.batchNorm)

        self.deconv1 = deconv(ngf * 1 + 2, ngf // 2, self.batchNorm)
        self.deconv0 = deconv(ngf // 2 + 2, ngf // 4, self.batchNorm)

        #>>>>>>>>>>>
        # self.inter_conv6 = i_conv(ngf * 32 + 2, ngf * 16, self.batchNorm)
        #<<<<<<<<<<<

        self.inter_conv5 = i_conv(ngf * 16 + 2, ngf * 8, self.batchNorm)
        self.inter_conv4 = i_conv(ngf * 8 + ngf * 4 + 2, ngf * 4, self.batchNorm)
        self.inter_conv3 = i_conv(ngf * 4 + ngf * 2 + 2, ngf * 2, self.batchNorm)

        self.inter_conv2 = i_conv(ngf * 1 + 2, ngf * 1, self.batchNorm)
        self.inter_conv1 = i_conv(ngf // 2 + 2, ngf // 2, self.batchNorm)
        self.inter_conv0 = i_conv(ngf // 4 + 2, ngf // 4, self.batchNorm)

        #>>>>>>>>>>>>>
        # self.predict_flow7 = predict_flow(ngf * 16)
        #<<<<<<<<<<<<<

        self.predict_flow6 = predict_flow(ngf * 16)
        self.predict_flow5 = predict_flow(ngf * 8)
        self.predict_flow4 = predict_flow(ngf * 4)
        self.predict_flow3 = predict_flow(ngf * 2)
        self.predict_flow2 = predict_flow(ngf * 1)
        self.predict_flow1 = predict_flow(ngf // 2)
        self.predict_flow0 = predict_flow(ngf // 4)

        # >>>>>>>>>>>>>
        # self.upsampled_flow7_to_6 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        #<<<<<<<<<<<<<

        # self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        # self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        # self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        # self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        # self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        # self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        # >>>>>>>>>>>>>
        # self.upsampled_flow7_to_6 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        #<<<<<<<<<<<<<

        self.upsampled_flow6_to_5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsampled_flow5_to_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsampled_flow4_to_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsampled_flow3_to_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsampled_flow2_to_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsampled_flow1_to_0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, inputxy):
        out_conv0 = self.conv0(inputxy)  # output: B*(ngf)*256*256
        out_conv1 = self.conv1_1(self.conv1(out_conv0))  # output: B*(ngf*2)*128*128
        out_conv2 = self.conv2_1(self.conv2(out_conv1))  # output: B*(ngf*2)*64*64

        out_conv3 = self.conv3_1(self.conv3(out_conv2))  # output: B*(ngf*4)*32*32
        out_conv4 = self.conv4_1(self.conv4(out_conv3))  # output: B*(ngf*8)*16*16
        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # output: B*(ngf*8)*8*8
        out_conv6 = self.conv6_1(self.conv6(out_conv5))  # output: B*(ngf*16)*4*4

        # >>>>>>>>>>>>>
        # out_conv7 = self.conv7_1(self.conv7(out_conv6))  # output: B*(ngf*16)*2*2
        #
        # flow7 = self.predict_flow7(out_conv7)  # output: B*2*2*2
        # flow7_up = self.upsampled_flow7_to_6(flow7)
        # out_deconv6 = self.deconv6(out_conv7)
        #
        # concat6 = torch.cat((out_conv6, out_deconv6, flow7_up), 1)
        # out_interconv6 = self.inter_conv6(concat6)
        # flow6 = self.predict_flow6(out_interconv6)  # output: B*2*4*4
        # <<<<<<<<<<<<<

        flow6 = self.predict_flow6(out_conv6)  # output: B*2*4*4
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        # concat5 = torch.cat((out_conv5, out_deconv5), 1)  #me
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)  # output: B*2*8*8

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        # concat4 = torch.cat((out_conv4, out_deconv4), 1)  #me
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)  # output: B*2*16*16
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        # concat3 = torch.cat((out_conv3, out_deconv3), 1)  #me
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)  # output: B*2*32*32
        flow3_up = self.upsampled_flow3_to_2(flow3)  # output: B*2*64*64
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_deconv2, flow3_up), 1)
        # concat2 = out_deconv2  #me
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)  # output: B*2*64*64
        flow2_up = self.upsampled_flow2_to_1(flow2)  # output: B*2*128*128
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_deconv1, flow2_up), 1)
        # concat1 = out_deconv1  #me
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)  # output: B*2*128*128
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((out_deconv0, flow1_up), 1)
        # concat0 = out_deconv0  #me
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)  # output: B*2*256*256

        # 256*256,128*128,64*64
        return flow0  # ,flow1,flow2
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='zero', use_activation='tanh'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        # activation = nn.ReLU(True)
        activation = nn.LeakyReLU(0.2)

        # model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=1), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
            model += [TransposeConv2dLayer(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=1, scale_factor=2),
                       norm_layer(int(ngf * mult / 2)), activation]

        if use_activation[:4] == 'tanh':
            try:
                weight = float(use_activation.split('_')[-1])
            except:
                weight = 1.0
            model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
            model += [get_norm_layer('line', weight), nn.Tanh()]
        elif use_activation == 'none':
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]  # , nn.Tanh()]
        else:
            print('error activation type!!!')
            exit()

        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)

class GlobalGeneratorPaper(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', use_activation='tanh'):
        assert (n_blocks >= 0)
        super(GlobalGeneratorPaper, self).__init__()
        activation = nn.ReLU(True)
        print(norm_layer)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
            model += [TransposeConv2dLayer(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=1, scale_factor=2),
                      norm_layer(int(ngf * mult / 2)), activation]

        if use_activation[:4] == 'tanh':
            try:
                weight = float(use_activation.split('_')[-1])
            except:
                weight = 1.0
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model += [get_norm_layer('line', weight), nn.Tanh()]
        elif use_activation == 'none':
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]#, nn.Tanh()]
        else:
            print('error activation type!!!')
            exit()

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, scale_factor=2):
        super(TransposeConv2dLayer, self).__init__()
        self.scale_factor = scale_factor
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')  # nearest bilinear
        x = self.conv2d(x)
        return x

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ArcLoss(nn.Module):
    def __init__(self):
        super(ArcLoss, self).__init__()
        self.loss = torch.nn.CosineSimilarity()


    def forward(self, x, y):

        return 1 - self.loss(x,y)
