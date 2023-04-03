import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_module import BaseModule, ResNet_Block, Discriminator, get_D_norm_layer, init_weights
from .stylegan.model import PixelNorm, EqualLinear
from .motion_network import define_G

from util import *
from util.helper import Glint_log as LOG
from collections import OrderedDict
from math import ceil

class SymmetricWarpingNet(BaseModule):
    def __init__(self, 
                 model_path = None,
                 resume_path = None,
                 config = None,
                 force_load_epoch = None
                ):
        
        super(SymmetricWarpingNet, self).__init__(model_path, resume_path, config, force_load_epoch)
        self.build_network()
        self.enable_backward()
        if resume_path is None:
            if self.config.initial_config.is_init:
                for k, v in self.net.named_children():
                    if k not in self.config.init_black_list:
                        init_weights(v, init_type = self.config.initial_config.init_type,gain = self.config.initial_config.init_gain)
        else:
            self.load_model(epoch = None, black_list = self.config.partial_load, force_load = self.config.force_load)

    def default_config(self):
        _key = ["ngf", "initial_config", "force_load", 
                "partial_load", "lr_mul", "use_kaiser_kernel",
                "equivalent"]
        _value = [64, {"init_type": "normal", "init_gain":0.02, "is_init": False}, False,
                 [], 1, False,
                 False]        
        return dict(zip(_key, _value))

    def convert_state_dict(self, model, state_dict):
        """modulated conv2d using static kernel.
        """
        from collections import OrderedDict
        oddict = OrderedDict()
        for key, value in model.state_dict().items():
            
            if "conv.conv.conv.weight" in key:
                replace_key = key.replace("conv.conv.conv.weight", "conv.conv.weight")
                oddict[key] = state_dict[replace_key].squeeze(0)
            elif "norm.blur.conv.weight" in key:
                replace_key = key.replace("norm.blur.conv.weight", "norm.kernel")
                oddict[key] = torch.flip(state_dict[replace_key],[0,1]).unsqueeze(0).unsqueeze(0)
            elif "blur.conv.weight" in key:
                replace_key = key.replace("blur.conv.weight", "kernel")
                oddict[key] = torch.flip(state_dict[replace_key],[0,1]).unsqueeze(0).unsqueeze(0)
            elif "conv.weight" in key:
                replace_key = key.replace("conv.weight", "weight")
                oddict[key] = state_dict[replace_key]
            elif "conv.bias" in key:
                replace_key = key.replace("conv.bias", "bias")
                oddict[key] = state_dict[replace_key]
            else:
                replace_key = key
                oddict[key] = state_dict[replace_key]
        return oddict

    @staticmethod
    def get_encoder(arch):
        gblocks = []
        for l_id in range(1, len(arch["layers_enc"])):
            gblock = ResNet_Block(
                arch["layers_enc"][l_id - 1],
                arch["layers_enc"][l_id],
                (arch["downsample"][l_id - 1]),
                arch["equivalent"]
            )
            gblocks += [gblock]

        return nn.Sequential(*gblocks)

    @staticmethod
    def get_decoder(arch):
        eblocks = []
        for l_id in range(1, len(arch["layers_dec"])):
            eblock = ResNet_Block(
                arch["layers_dec"][l_id - 1],
                arch["layers_dec"][l_id],
                arch["upsample"][l_id - 1],
                arch["equivalent"]
            )
            eblocks += [eblock]

        return nn.Sequential(*eblocks)

    @staticmethod
    def get_mapping(arch, lr_mul):
        style_dim = arch["style_dim"]
        mblocks = [PixelNorm()]
        for _ in range(2):
            for _ in range(len(arch["layers_dec"])):
                mblocks.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mul, activation='fused_lrelu'))
        return nn.Sequential(*mblocks)

    def get_arch(self):
        return self.arch

    def build_network(self):
        in_channels = 3
        arch = {
            "layers_enc": [
                in_channels,
                self.config.ngf // 2,
                self.config.ngf // 2,
                self.config.ngf // 2,
                self.config.ngf,
                self.config.ngf,
                self.config.ngf,
                self.config.ngf,
                64 + 1,
            ],
            "downsample": [
                'False',
                'False',
                'False',
                'False',
                'False',
                'False',
                'False',
                'False',
            ],
            "layers_dec": [
                64,
                self.config.ngf,
                self.config.ngf * 2,
                self.config.ngf * 4,
                self.config.ngf * 4,
                self.config.ngf * 2,
                self.config.ngf * 2,
                self.config.ngf * 2,
                3,
            ],
            "upsample": [
                "False",
                "Down",
                "Down",
                "False",
                "Up",
                "Up",
                "False",
                "False",
            ],
            "style_dim": 512,
            "use_kaiser_kernel": self.config.use_kaiser_kernel,
            "equivalent": self.config.equivalent
        }
        if self.config.use_kaiser_kernel:
            coef = 2
            content = arch['downsample']
            for i in range(len(content)):
                content[i] = content[i] + "+kaiser" + f"+{coef}" + "+sinc"
            arch['downsample'] = content

            content = arch['upsample']
            count = 0
            coef_weight = 2 ** -0.3
            coef_cur = coef

            content[0] = content[0] + "+kaiser" + f"+{coef_cur}" + "+sinc"
            for i in range(1,len(content)):
                value = content[i]
                if value == "Down":
                    count += 1
                elif value == "Up":
                    count -= 1
                coef_cur = coef * (coef_weight ** count)
                content[i] = content[i] + "+kaiser" + f"+{coef_cur}" + "+sinc"
            arch['upsample'] = content

        self.arch = arch

        self.net['encoder'] = SymmetricWarpingNet.get_encoder(arch)
        self.net['decoder'] = SymmetricWarpingNet.get_decoder(arch)
        self.net['mapping'] = SymmetricWarpingNet.get_mapping(arch, self.config.lr_mul)
        self.symmetric_warping = SymmetricSplatting() 
        self.act = nn.Tanh()

    def feature_encode(self, x, random_from_outside):
        feature_x = x
        latent_noise_vector = random_from_outside #self.net['mapping'](random_from_outside)
        for name, module in self.net['encoder'].named_children():
            feature_x = module(feature_x, latent_noise_vector)
        return feature_x, latent_noise_vector

    def feature_decode(self, latent_noise_vector, feature_warped):
        for name, module in self.net['decoder'].named_children():
            feature_warped = module(feature_warped, latent_noise_vector)
        return feature_warped

    def forward(self, 
                x, 
                y,
                flow,
                t: int,
                N: int,
                random_from_outside = None,
                use_cache = False,
                use_fast_version = False,
                ):

        n, _, h, w = x.shape
        h_, w_ = int(ceil(h / 4) * 4), int(ceil(w / 4) * 4)
        h_f, w_f = flow.shape[2:]
        flow = F.interpolate(flow, (h_, w_))
        target_flow_forward = flow.clone()
        target_flow_forward = clip_flow(target_flow_forward, compute_meshgrid(target_flow_forward.shape, False).to(target_flow_forward))
        target_flow_backward = -1 * flow.clone()
        target_flow_backward = clip_flow(target_flow_backward, compute_meshgrid(target_flow_backward.shape, False).to(target_flow_backward))

        if use_fast_version:
            flow = F.interpolate(flow, scale_factor = 0.25)

        if use_cache:
            euler_func = EulerIntegration(N, use_cache)
            # eulerian integer.
            for i in range(n):
                target_flow_forward_o, target_flow_backward_o = euler_func(flow[i:i+1, ...], t)
                if use_fast_version:
                    target_flow_forward[i,...] = F.interpolate(target_flow_forward_o, scale_factor = 4)
                    target_flow_backward[i,...] = F.interpolate(target_flow_backward_o, scale_factor = 4)
                target_flow_forward[i,0,...] *= (w_)
                target_flow_forward[i,1,...] *= (h_)
                target_flow_backward[i,0,...] *= (w_)
                target_flow_backward[i,1,...] *= (h_)
            t = torch.ones((1)) * t
            t = t.to(flow)
            N = torch.ones((1)) * N
            N = N.to(flow)
        else:
            t = torch.ones((1)).to(x) * t
            t = t.to(flow)
            N = torch.ones((1)) * N
            N = N.to(flow)
            # eulerian integer.
            for i in range(n):
                t_i = t[i]
                N_i = N[i]
                target_flow_forward[i,...] = euler_integration(flow[i:i+1,...], t_i)
                target_flow_backward[i,...] = euler_integration(-1 * flow[i:i+1, ...], N_i - t_i)
                target_flow_forward[i,0,...] *= (w)
                target_flow_forward[i,1,...] *= (h)
                target_flow_backward[i,0,...] *= (w)
                target_flow_backward[i,1,...] *= (h)
        latent_noise_vector = self.net['mapping'](torch.randn((x.size(0), 512)).to(x) if random_from_outside is None else random_from_outside)

        if y is not None:
            feature_x, feature_y = x, y
            feature_x = F.interpolate(feature_x, (h_, w_))
            feature_y = F.interpolate(feature_y, (h_, w_))

            for name, module in self.net['encoder'].named_children():
                feature_x = module(feature_x, latent_noise_vector)
                feature_y = module(feature_y, latent_noise_vector)
            metric_x = feature_x[:,-1:,:,:]
            metric_y = feature_y[:,-1:,:,:]
        else:
            feature_x = x
            feature_x = F.interpolate(feature_x, (h_, w_))
            for name, module in self.net['encoder'].named_children():
                feature_x = module(feature_x, latent_noise_vector)
            feature_y = feature_x
            metric_y = metric_x = feature_x[:,-1:,:,:]

        feature_warped = self.symmetric_warping(feature_x[:,:-1, :, :], target_flow_forward, metric_x, \
                                                feature_y[:,:-1, :, :], target_flow_backward, metric_y, \
                                                t, N)

        for name, module in self.net['decoder'].named_children():
            feature_warped = module(feature_warped, latent_noise_vector)
    
        if self.training:
            return F.interpolate(self.act(feature_warped), (h, w)), target_flow_forward, target_flow_backward, metric_x, metric_y
        else:
            return F.interpolate(self.act(feature_warped), (h, w))

class encoderInWarpNet(nn.Module):
    def __init__(self, net_encoder):
        super().__init__()
        self.net = net_encoder

    
    def forward(self, x, latent_noise_vector):
        if latent_noise_vector is None:
            return self.net(x)
        feature_x = x
        for name, module in self.net.named_children():
            feature_x = module(feature_x, latent_noise_vector)
        return feature_x

class StyleGanDiscriminator(BaseModule):
    def __init__(self, 
                 model_path = None,
                 resume_path = None,
                 config = None,
                 force_load_epoch = None,
                ):
        super(StyleGanDiscriminator, self).__init__(model_path, resume_path, config, force_load_epoch)
        self.build_network()
        self.enable_backward()
        if resume_path is None:
            if self.config.initial_config.is_init:
                init_weights(self.net, init_type = self.config.initial_config.init_type,gain = self.config.initial_config.init_gain)
        else:
            self.load_model(resume_path, black_list = self.config.partial_load)

    def default_config(self):
        _default_config_key = ['style_gan_path','initial_config','partial_load',
                               'size'
                              ]
        _default_config_value = ['weights/style_gan_d.pth',{"init_type": "normal", "init_gain":0.02, "is_init": False}, [],
                                 256
                                ]
        return dict(zip(_default_config_key, _default_config_value))

    def build_network(self):
        self.net = nn.ModuleDict()
        d = Discriminator(self.config.size)
        self.net['Discriminator'] = d
    
    def forward(self, x, feature_match = False):
        if feature_match:
            features = []
        else:
            features = None
        conv_y = x
        for name, module in self.net['Discriminator'].convs.named_children():
            conv_y = module(conv_y)
            if feature_match and conv_y.size(2) >= 32:
                features.append(conv_y)
        y = self.net['Discriminator'].get_conv_std(conv_y)
        final_y = self.net['Discriminator'].final_conv(y)
        y = final_y.view(x.size(0), -1)
        return self.net['Discriminator'].final_linear(y), features

class MultiscaleDiscriminator(BaseModule):
    def __init__(self, 
                 model_path = None,
                 resume_path = None,
                 config = None,
                 force_load_epoch = None,
                ):
        super(MultiscaleDiscriminator, self).__init__(model_path, resume_path, config, force_load_epoch)
        self.build_network()
        self.enable_backward()
        if self.config.initial_config.is_init:
            init_weights(self.net, init_type = self.config.initial_config.init_type,gain = self.config.initial_config.init_gain)
        if resume_path is not None:
            self.load_model(resume_path, black_list = self.config.partial_load)

    def default_config(self):
        _default_config_key = ['initial_config','partial_load', "norm_D",
                               'ndf', 'D_num'
                              ]
        _default_config_value = [{"init_type": "xavier", "init_gain":0.02, "is_init": True}, [], "spectralinstance",
                                 64, 2
                                ]
        return dict(zip(_default_config_key, _default_config_value))

    def build_single_discriminator(self):
        norm_layer = get_D_norm_layer(self.config.norm_D)
        kw = 4
        n_layers_D = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = self.config.ndf
        input_nc = 3

        sequence = [
                nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, False),]
        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [
                    norm_layer(
                        nn.Conv2d(
                            nf_prev,
                            nf,
                            kernel_size=kw,
                            stride=stride,
                            padding=padw,
                        )
                    ),
                    nn.LeakyReLU(0.2, False),]
            
        sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]
        
        return nn.Sequential(*sequence)

    def build_network(self):
        for i in range(self.config.D_num):
            self.net["{}".format(i)] = self.build_single_discriminator()

    def forward(self, x, feature_match = False):
        if feature_match:
            features = []
        else:
            features = None
        results = []
        input_x = x
        for i in range(self.config.D_num):
            conv_y = input_x
            if feature_match:
                feature_list = []
            for name, module in self.net["{}".format(i)].named_children():
                conv_y = module(conv_y)
                if feature_match:
                    feature_list.append(conv_y)
            input_x = F.avg_pool2d(input_x,kernel_size=3,stride=2,padding=[1, 1],count_include_pad=False,)
            results.append(conv_y)
            if feature_match:
                features.append(feature_list)
        return results, features

class MotionEstimationNet(BaseModule):
    def __init__(self, 
                 model_path = None,
                 resume_path = None,
                 config = None,
                 force_load_epoch = None,
                ):
        super(MotionEstimationNet, self).__init__(model_path, resume_path, config, force_load_epoch)
        self.build_network()
        self.enable_backward()
        if resume_path is None:
            if self.config.initial_config.is_init:
                init_weights(self.net, init_type = self.config.initial_config.init_type,gain = self.config.initial_config.init_gain)
        else:
            self.load_model(resume_path, black_list = self.config.partial_load)

    def default_config(self):
        _default_config_key = ['motion_net_path','initial_config','partial_load',
                               'ngf', 'netG', 'n_downsample_global', 
                               'n_blocks_global', 'n_blocks_local', 'n_local_enhancers',
                               'norm','local_net_path', 'adaptive_global'
                              ]
        _default_config_value = [None, {"init_type": "normal", "init_gain":0.02, "is_init": False}, [],
                                 32, 'global', 4,
                                 9, 3, 1,
                                 'instance', None, True
                                ]
        return dict(zip(_default_config_key, _default_config_value))

    def build_network(self):
        self.net = nn.ModuleDict()
        self.net['motion_estimation'] = define_G(3, 2, self.config.ngf, self.config.netG, 
                                      self.config.n_downsample_global, self.config.n_blocks_global, self.config.n_local_enhancers, 
                                      self.config.n_blocks_local, self.config.norm, adaptive_global = self.config.adaptive_global)
        
        if self.config.motion_net_path is not None:
            motion_state_dict = torch.load(self.config.motion_net_path)
            
            # to fix bug in module.
            shadow_state_dict = OrderedDict()
            for k, v in motion_state_dict.items():
                name = k
                if 'module' in name:
                    name = name.replace('module.', '')
                shadow_state_dict[name] = v
            self.net["motion_estimation"].load_state_dict(shadow_state_dict)
            LOG("info")(f"load motion estimation {self.config.motion_net_path} successfully.")

        if self.config.local_net_path is not None:
            motion_state_dict = torch.load(self.config.local_net_path)
            self.net["motion_estimation"].model.load_state_dict(motion_state_dict)
            LOG("info")(f"load motion estimation {self.config.local_net_path} successfully.")

    def forward(self, x):
        x = torch.flip(x,[1])
        return self.net["motion_estimation"](x)

        





