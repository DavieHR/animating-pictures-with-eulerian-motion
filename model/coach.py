
import os
import yaml
from functools import partial
from easydict import EasyDict as edict
from util.helper import Glint_log as LOG

import torch
import data

from . import networks as NR
from util import loss as L
from util.parallel import MultiGPU
from util.vis import Visor

from torch.optim import Adam 
from pytorch_ranger import Ranger
from torch.optim import lr_scheduler

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cycledrop':
        def learning_tune_func(epoch):
            cycle = math.floor(1 + epoch / (2 * opt.lr_step))
            x = abs(epoch / opt.lr_step - 2 * cycle + 1)
            cyclical_learning = 1 +  (opt.max_lr - opt.lr) * max(0, (1-x))
            return cyclical_learning
        #lambda_rule = lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_tune_func)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    return scheduler

class Coach:

    def __init__(self, 
                 opt
                 ):

        data_func = getattr(data, opt.data_type)
        self.train_data = data_func("train",opt)
        #self.test_data = data_func("test", opt)
        self.device = "cuda:0" if opt.gpu_nums > 0 else "cpu"
        self.gpu_ids = range(opt.gpu_nums)
        self.visor = Visor(opt.tensorboard)
        self.network_names = {}
        self.opt = opt
        self.total_steps = 0
        self.type_par = "pytorch"

    def load_config(self):
        script_path = self.opt.script_path
        if script_path == '':
            return None
        config_path = os.path.join(script_path, 'config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'rb') as f:
                    config = edict(yaml.load(f))
                LOG("INFO")("config yaml is {}".format(config))
            except Exception as e:
                LOG("INFO")("config access fail, {}".format(e))
                config = None
        else:
            LOG("WARN")("No config exists")
            config = None
        return config
        
    def register_network(self):

        networks = {}
        snapshots = self.opt.snapshots
        
        for name, k in self.network_names.items():
            if k == '':
                continue
            if self.config is None:
                _net_config = None
            else:
                _net_config = getattr(self.config, name) if hasattr(self.config, name) else None
            model_path = os.path.join(snapshots, name) if snapshots is not None else None
            resume_total_path = self.opt.resume_path if self.opt.resume_path is not None else None
            resume_path = os.path.join(resume_total_path, name) if resume_total_path is not None else None
            instance_net = getattr(NR, k)(model_path, resume_path, config = _net_config)
            instance_net = MultiGPU(self.gpu_ids, self.type_par)(instance_net)              
            LOG("INFO")("network: {} initialization complete".format(name))
            networks[name] = instance_net
        return edict(networks)

    def register_loss(self):
        """register loss module from yaml.
        """
        loss_config = getattr(self.config, 'Losses')
        loss = {}
        for k,_loss in loss_config.items():
            _loss_coef = _loss.coef if hasattr(_loss, 'coef') else {}
            _loss_weight = _loss.weight
            instance_loss = getattr(L, k)()
            #instance_loss = MultiGPU(self.gpu_ids)(instance_loss)
            _loss_components = {'loss': instance_loss, 'weight': _loss_weight, 'other':_loss_coef}
            loss[k] = _loss_components
        return edict(loss)
    
    def register_optimizer(self, merge_parameter):
        """register optimizer from yaml.
        """
        opt_config = getattr(self.config, 'Optimizer')
        opt = {}
        if merge_parameter is not None:
            ignore_name = merge_parameter.split("+")[-1]
            merge_name = merge_parameter.split("+")[0]
        else:
            ignore_name = ""
            merge_name = ""
        for name, _ in self.network_names.items():
            if name == ignore_name:
                continue
            optimizer_param = {'params':self.net[name].parameters()}
            optimizer_param['lr'] = self.opt.learning_rate
            if self.opt.learning_rate_coeffs is not None:
                for coef_pair in self.opt.learning_rate_coeffs:
                    _key_diff_lr = coef_pair.split(',')[0]
                    coeff = float(coef_pair.split(',')[1])
                    if _key_diff_lr == name:
                        optimizer_param['lr'] *= coeff
                        break
            optimizer_param = [optimizer_param]
            if merge_name == name:
                lr = self.opt.learning_rate
                if self.opt.learning_rate_coeffs is not None:
                    for coef_pair in self.opt.learning_rate_coeffs:
                        _key_diff_lr = coef_pair.split(',')[0]
                        coeff = float(coef_pair.split(',')[1])
                        if _key_diff_lr == ignore_name:
                            lr *= coeff
                            break
                optimizer_param += [{"params":self.net[ignore_name].parameters(),"lr": lr}]
            opt[name] = eval(opt_config.type)(optimizer_param, betas = (opt_config.beta1, 0.99))
            lr = opt[name].param_groups[0]['lr']
            self.visor["TextLearningRate_" + name] = lr
        self.visor(0)
        return edict(opt)

    def register_scheduler(self):
        sch = {}
        for name, v in self.optimizers.items():
            sch[name] = get_scheduler(v, self.opt)
        return edict(sch)

    # print network information
    def print_networks(self, verbose=False):
        LOG("info")('---------- Networks initialized -------------')
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self.net, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    LOG("debug")(net)
                LOG("info")('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        LOG("info")('-----------------------------------------------')

    def save_networks(self, epoch):
        for k, v in self.net.items():
            LOG("Save ---- epoch {} ---- {} models".format(epoch, k))
            v.module.save_model(epoch)
    
    def update_optimizers(self, epoch):
        for k, scheduler in self.schedulers.items():
            scheduler.step()
            lr = self.optimizers[k].param_groups[0]['lr']
            self.visor["TextLearningRate_" + k] = lr
        self.visor(epoch)






