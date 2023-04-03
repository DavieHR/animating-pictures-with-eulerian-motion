import os
import sys
sys.path.insert(0, os.getcwd())

import torch
from model.networks import SymmetricWarpingNet, StyleGanDiscriminator
from util.helper import Glint_log as LOG
from easydict import EasyDict as edict

def test_symmetric_warp_net():

    device = "cuda:0"

    """Class Instance
    """
    net_gen = SymmetricWarpingNet().to(device)
    config = edict({"size": 256})
    net_dis = StyleGanDiscriminator(config = config).to(device)
    
    LOG("info")("Class Instance Successful. ")


    """Inference.
    """
    x = torch.randn((1,3,256,256)).to(device)
    y = torch.randn((1,3,256,256)).to(device)
    flow = torch.randn((1,2,256,256)).to(device)
    z, _, _, _, _ = net_gen(x,y,flow,3,50)
    print(z)
    z_dot = net_dis(z)
    LOG("info")("Inference Successful. ")

    """Save Model.
    """
    net_gen.save_model(0)
    LOG("info")("Save Model Successful. ")

    """Load Model
    """
    net_gen.load_model()
    LOG("info")("Load Model Successful. ")

    net_gen.DestroyNet()
    LOG("info")("Destroy Model Successful. ")






