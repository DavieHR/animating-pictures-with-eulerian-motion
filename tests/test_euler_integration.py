import os
import sys
sys.path.insert(0, os.getcwd())
import cv2
import imageio
import numpy as np

import torch
from data.dataset import readFlow
from util.euler_integration import euler_integration
from util.flow_utils import flow2img


if __name__ == '__main__':

    argv = sys.argv

    flow = readFlow(argv[1])
    if not os.path.exists(argv[2]):
        os.makedirs(argv[2])
    save_path = argv[2]
    flow_vis = flow2img(flow)
    flow = torch.from_numpy(flow)
    flow = flow.permute((2,0,1))
    flow = flow.to(torch.float32)
    h_flow, w_flow = flow.size(1), flow.size(2)
    cv2.imwrite(os.path.join(save_path,"1.png"), flow_vis)
    flow = flow.unsqueeze(0)
    with imageio.get_writer(os.path.join(save_path, "tutu_forward_correct.mp4"), fps = 10, macro_block_size = None) as writer:
        for i in range(1, 60):
            print(i)
            i_tensor = torch.ones((1)) * 1
            if i == 1:
                flow_euler, cordinate = euler_integration(flow, i_tensor, None, None)
            else:
                flow_euler, cordinate = euler_integration(flow, i_tensor, cordinate, flow_euler) 
            ##flow_euler, _ = euler_integration(flow, i_tensor)
            flow_vis = flow2img(flow_euler.squeeze().detach().permute((1,2,0)).numpy().squeeze())
            writer.append_data(np.uint8(flow_vis))
