import os
import sys
import json
from easydict import EasyDict as edict
from collections import OrderedDict
import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import init
import torch.nn.utils.spectral_norm as spectral_norm

from functools import partial
from .stylegan.model import Downsample, FusedLeakyReLU, PixelNorm, EqualConv2d, \
                            EqualLinear, NoiseInjection, StyledConv, Discriminator, \
                            ModulatedConv2d, Upsample, Upsample_kaiser, Downsample_kaiser, \
                            EquivalentEqualConv2d, EquivalentModulatedConv2d
                            
from util.helper import Glint_log as LOG

class BaseModule(nn.Module):
    """Base Module as all network.
       It will include dump configuration fucntion,
       load model weight, load configuration, module visualization,
       convert to libtorch c++ model code, etc....
       Attributes: 
            model_path: str, the model path.
            config: dict, the model configuration.
            
    """
    def __init__(self, model_path, resume_path, config = None, force_load_epoch = None):
        super(BaseModule, self).__init__()
        if model_path is None:
            model_path = "default_net"
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok = True)
        self.model_path = model_path
        self.config_path = os.path.join(model_path,'config.json')
        self.resume_path = resume_path if resume_path is not None else "default_net"
        
        self.force_load_epoch = force_load_epoch
        self.config_resume_path = os.path.join(resume_path,'config.json') if resume_path is not None else None
        self.config = self.config_parse(self.config_resume_path) if config is None else config
        self._merge_config_with_default()
        LOG("info")("config is {}".format(self.config))
        self.net = nn.ModuleDict()

    def _merge_config_with_default(self):
        config_default = edict(self.default_config())
        for k, v in config_default.items():
            if k not in self.config:
                self.config[k] = v

    def _temp_config(self):
        """you should ref by yourself.
           it is a list
        """
        return []

    def config_parse(self, config_path):
        """parse configuration from config path.
        """
        if config_path is not None and os.path.exists(config_path):
            return self._load_config(config_path)
        return edict(self.default_config())
    
    def default_config(self):
        """ parent function
        """

    def build_network(self):
        """ parent function
        """
        pass

    def convert_state_dict(self, model, state_dict):
        
        return state_dict

    def load_model(self, epoch = None, black_list = [], force_load = False):
        if not os.path.exists(self.resume_path):
            LOG("info")("please ref your resume path parent path.")
            return

        file_list = sorted(list(filter(lambda x: '.pth' in x or '.pt' in x,os.listdir(self.resume_path))), key=lambda d:int(d.split('.')[0]))
        epoch = self.force_load_epoch
        _load_path = os.path.join(self.resume_path, file_list[-1] if epoch is None else file_list[epoch])
        _file_name = file_list[-1] if epoch is None else file_list[epoch]
        dict_model = torch.load(_load_path)
        if "equivalent" in self.config and self.config["equivalent"]:
            dict_model = self.convert_state_dict(self.net, dict_model)
        if len(black_list) == 0:
            self.net.load_state_dict(dict_model, (not force_load))
        else:
            for name, module in self.net.named_children():
                if name not in black_list:
                    filtered_dict = OrderedDict()
                    for k, v in dict_model.items():
                        if name in k:
                            new_name = k.replace('net.' + name + '.', '')
                            filtered_dict[new_name] = v
                    self.net[name].load_state_dict(filtered_dict, False)

        LOG("INFO")("load the latest model {} from {}".format(_file_name, self.resume_path))

    def save_model(self, epoch):
        """ save model.
        """
        if isinstance(self.net, nn.DataParallel):
            save_net = self.net.module
        else:
            save_net = self.net
        _model_dir = os.path.join(self.model_path, '{}.pth'.format(epoch))
        torch.save(save_net.state_dict(), _model_dir)
        self.dump_config()


    def dump_config(self):
        """ dump configuration of this module.
            Returns:
                   None
            Raises:
                   Exception, Runtime.
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f)
        except Exception as e:
            LOG("ERROR")("dump config error! please check error feedback {0:^8s}".format(e)) 
            raise

    # TODO: load config from yaml file.
    # Author: wanghaoran@glint.com
    # Time: 2020-0731
    def _load_config(self, path):
        """load configuration from path.
           Args: 
                path, str, your config path.
           Returns:
                configuration, dict.
           Raise:
                Exception, Runtime error.
        """
        try:
            with open(path, 'rb') as f:
                config = json.load(f)

            config_dict = edict(config)
            _temp_config = self._temp_config()
            _default_config = self.default_config()
            for key in _temp_config:
                config_dict[key] = _default_config[key]
            return config_dict
        except Exception as e:
            LOG("ERROR")("load config error! please check error feedback {0:^8s}".format(e)) 
            raise

    def instance_network(self):
        """parent.
        """
        pass

    def convert_jit_libtorch(self, path):
        """convert python model to libtorch model.
        """
        net_jit = torch.jit.script(self.instance_network(self.net))
        torch.jit.save(net_jit, path)
 
    def enable_backward(self):
        for p in self.net.parameters():
            p.requires_grad = True

    def disable_backward(self):
        for p in self.net.parameters():
            p.requires_grad = False

    def DestroyNet(self):
        LOG("WARNING")("REMOVE ALL MODEL SAVE IN YOUR PC.")
        import shutil
        shutil.rmtree(self.model_path)

class NoiseStyledConv(nn.Module):
    def __init__(self, in_c, in_o, k, s_d, activate = True, kaiser_kernel = None, equivalent = False):
        super(NoiseStyledConv, self).__init__()
        self.conv = StyledConv(in_c, in_o, k, s_d, use_activate = activate, kaiser_kernel = kaiser_kernel, equivalent = equivalent)
        #self.linear_layer = EqualLinear(s_d, s_d, lr_mul = 0.01)
        #self.scale = 1 / math.sqrt(s_d ** 2)
        #self.style_dim = s_d

    def forward(self, x, latent_noise_vector):
        #latent_noise_vector = self.linear_layer(torch.randn((x.size(0), self.style_dim)).to(x) * self.scale)
        return self.conv(x, latent_noise_vector)

class ResNet_Block(nn.Module):
    def __init__(self, in_c, in_o, downsample=None, equivalent = False):
        super().__init__()
        
        if not isinstance(downsample, str):
            downsample = str(downsample)
        if downsample == "Down":
            norm_downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) #Downsample([1,3,3,1], use_in_net_conv = equivalent) #nn.AvgPool2d(kernel_size=3, stride=2, padding=1) #Downsample([1,3,3,1], use_in_net_conv = equivalent) #nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            kernel = None
        elif downsample == "Up":
            norm_downsample = Upsample([1,3,3,1], use_in_net_conv = equivalent) #nn.Upsample(scale_factor=2, mode="bilinear")
            kernel = None
        elif 'Down+kaiser' in downsample:
            coef, func = float(downsample.split('+')[2]),  downsample.split('+')[3]
            norm_downsample = Downsample_kaiser(func, coef)
            kernel = [coef, func]
        elif 'Up+kaiser' in downsample:
            coef, func = float(downsample.split('+')[2]),  downsample.split('+')[3]
            norm_downsample = Upsample_kaiser(func, coef)
            kernel = [coef, func]
        else:
            if "kaiser" in downsample:
                coef, func = float(downsample.split('+')[2]),  downsample.split('+')[3]
                kernel = [coef, func]
            else:
                kernel = None
            norm_downsample = nn.Identity()

        conv_aa = NoiseStyledConv(in_c, in_o, 3, 512, kaiser_kernel = kernel, equivalent = equivalent)
        conv_ab = NoiseStyledConv(in_o, in_o, 3, 512, False, equivalent = equivalent)

        self.ch_a = nn.Sequential(
            conv_aa,
            conv_ab,

        )
        self.norm = norm_downsample
        conv_down = EqualConv2d if not equivalent else EquivalentEqualConv2d
        if downsample != 'False' or (in_c != in_o):
            conv_b = conv_down(in_c, in_o, 1, 1, 0)
            self.ch_b = nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = nn.Identity()

    def forward(self, x, latent_vector):
        x_a, x_b = x, x

        for name, module in self.ch_a.named_children():
            x_a = module(x_a, latent_vector)
        x_a = self.norm(x_a)
        x_b = self.ch_b(x)
        return x_a + x_b
        #return x_b

# copy from synsin normalization codes.
# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_D_norm_layer(norm_type="spectralinstance"):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, "out_channels"):
            return getattr(layer, "out_channels")
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith("spectral"):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len("spectral") :]

        if subnorm_type == "none" or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, "bias", None) is not None:
            delattr(layer, "bias")
            layer.register_parameter("bias", None)

        if subnorm_type == "batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)

        elif subnorm_type == "instance":
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError(
                "normalization layer %s is not recognized" % subnorm_type
            )

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


#copy from pixel2pixelHD repo.
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)
