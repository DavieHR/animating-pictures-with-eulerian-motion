import os
import sys
sys.path.insert(0, os.getcwd())


import cv2
import torch
import numpy as np
from model.networks import MotionEstimationNet
from util.helper import Glint_log as LOG
from easydict import EasyDict as edict
from util.flow_utils import readFlow, writeFlow, visulize_flow_file, flow2img

def test_motion_estimation_net():
    device = "cuda:0"

    """Class Instance
    """
    config = edict({"motion_net_path": None})
    net_motion = MotionEstimationNet().to(device)
    LOG("info")("Class Instance Successful. ")

    """Random Inference Test
    """
    x = torch.randn((1,3,256,256)).to(device)
    z = net_motion(x)
    LOG("info")("Random Inference Successful. ")
