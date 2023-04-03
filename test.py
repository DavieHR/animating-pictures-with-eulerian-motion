import os
import imageio
import cv2
import numpy as np
import torch
import time
from easydict import EasyDict as edict
from option import TestOption
from model.networks import *
from util.helper import Glint_log as LOG
from data.dataset import make_dataset
from util.flow_utils import readFlow, writeFlow, visulize_flow_file, flow2img
from util import euler_integration_once, SymmetricSplatting

def toTensor(x):
    return torch.from_numpy(x).permute((2,0,1)).unsqueeze(0) / 255.0

def fromTensor(x):
    return x.detach().cpu().squeeze().permute((1,2,0)).numpy()

if __name__ == '__main__':
    config = TestOption().get_parser()
    TestOption().print_options(config)
    device = "cuda:0" if config.gpu_nums > 0 else "cpu"
    
    config_gen = edict({"equivalent": False})
    GenNet = eval(config.Gen_name)(resume_path = os.path.join(config.resume_path, "Gen"), config = config_gen).to(device)
    GenNet.eval()
    if config.separation_emotion_path is None:
        EMotionNet = eval(config.EMotion_name)(resume_path = os.path.join(config.resume_path, "EMotion")).to(device)
    else:
        config_net = edict({"motion_net_path": config.separation_emotion_path})
        EMotionNet = eval(config.EMotion_name)(config = config_net).to(device)
    EMotionNet.eval()

    if os.path.isdir(config.input):
        datas = make_dataset(config.input)[0]
    else:
        datas = [config.input]
    for idx, data_path in enumerate(datas):

        LOG("info")(f"{idx} / {len(datas)}:  {data_path}")
        image = cv2.imread(data_path)[...,::-1]
        h,w = image.shape[:2]
        if h < w:
            image_720 = cv2.resize(image, (1280, 720))
        elif h > w:
            image_720 = cv2.resize(image, (720, 1280))
        else:
            image_720 = cv2.resize(image, (1280, 1280))
        image_tensor = (toTensor(image_720.copy()) * 2 - 1).to(device)
        if os.path.isdir(config.output):
            os.makedirs(config.output, exist_ok = True)
            data_name = os.path.basename(data_path).split('.')[0]
            save_dir = os.path.join(config.output, data_name + "_" + config.save_file_name)
        else:
            save_dir = config.output.split('.')[0] + config.save_file_name

        with imageio.get_writer(save_dir, fps = 30) as writer:
            if config.verbose:
                tic_motion = time.time()
            n,c,h,w = image_tensor.shape
            ff = image_tensor.clone()
            divisor = 2 ** 4
            w_, h_ = int(ceil(w / divisor) * divisor),int(ceil(h / divisor) * divisor)
            if h_ != h or w_ != w:
                ff = F.interpolate(image_tensor, (h_, w_))
            flow = EMotionNet(F.interpolate(ff, tuple(list(map(int, config.resolution)))))
            flow = flow.clamp(min=-1.0, max=1.0)
            flow = F.interpolate(flow, (h_, w_))
            flow *= config.intensity
            # flow = fromTensor(flow)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # flow_hard = flow.copy()
            # threshold = 0.0
            # flow_hard[np.abs(flow) > threshold] = 1.0
            # flow_hard[np.abs(flow) <= threshold] = 0.0
            # flow_hard = cv2.erode(flow_hard, kernel, iterations=6)
            # for _ in range(3):
            #     flow_hard = cv2.GaussianBlur(flow_hard, (5, 5), 0, 0)
            # flow = flow * flow_hard
            # # flow = flow_hard
            # flow = toTensor(flow).to(device) * 255.0

            # print('flow: ', flow.max().item(), flow.min().item())
            # print('flow abs: ', flow.abs().max().item(), flow.abs().min().item())

            #out_w = {'up': 0.0001, 'down': 0.0008}
            #flow_x = flow[:, 0:1]
            #flow_y = flow[:, 1:]
            #flow_x[flow_y > 0.0] *= out_w['down']
            #flow_x[flow_y < 0.0] *= out_w['up']
            #flow_y[flow_y > 0.0] *= out_w['down']
            #flow_y[flow_y < 0.0] *= out_w['up']
            #flow = torch.cat((flow_x, flow_y), 1)

            if h_ != h or w_ != w:
                flow = F.interpolate(flow, (h, w))
            #norm_factor = torch.cat((torch.ones((n,1,h,w)) * w_, torch.ones(n,1,h,w) * h_), dim = 1).to(flow)
            #flow = flow / norm_factor
            if config.verbose:
                toc_motion = time.time()
                LOG("DEBUG")(f"EMotionNet time elapse {toc_motion - tic_motion}")
            if config.verbose:
                LOG("DEBUG")("verbose mode: ON")
                debug_dir = os.path.join(os.path.dirname(save_dir), "debug") 
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                debug_file = os.path.join(debug_dir, os.path.basename(save_dir).split('.')[0] + "_flow_show.png")
                flow_to_image = flow2img(fromTensor(flow))[..., ::-1]
                cv2.imwrite(debug_file, flow_to_image)

            if config.verbose:
                tic_gen = time.time()

            flow_forward = torch.zeros_like(flow)
            flow_backward = torch.zeros_like(flow)
            N = config.frames
            N = torch.ones((1)) * N
            N = N.to(flow)
            n, _, h, w = image_tensor.shape
            h_, w_ = int(ceil(h / 4) * 4), int(ceil(w / 4) * 4)
            flow = F.interpolate(flow, (h_, w_))
            image_tensor = F.interpolate(image_tensor, (h_, w_))
            h_f, w_f = flow.shape[2:]
            tensor_hw = torch.cat((torch.ones((n,1,h_, w_)).to(flow) * w_, torch.ones((n,1,h_, w_)).to(flow) * h_), dim=1)
            flow = flow * tensor_hw
            # first integration.
            forward_flows = []
            backward_flows = []
            for t in range(config.frames):
                flow_forward = euler_integration_once(flow, flow_forward)
                forward_flows.append(flow_forward)

            for t in range(config.frames):
                flow_backward = euler_integration_once(-1 * flow, flow_backward)
                backward_flows.append(flow_backward)

            start_total_time = time.time()
            for t in range(config.frames):
                flow_forward = forward_flows[t]
                flow_backward = backward_flows[config.frames - t - 1]
                t = torch.ones((1)) * t
                t = t.to(flow)
                h, w = image_tensor.shape[2:]
                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    random = torch.randn(1, 512, device = device)
                    feature_x, latent = GenNet.feature_encode(image_tensor, random)
                    feature_y = feature_x.clone()
                    metric_x = feature_x[:,-1:,:,:]
                    metric_y = feature_y[:,-1:,:,:]
                    feature_warped = SymmetricSplatting()(feature_x[:,:-1, :, :], flow_forward, metric_x, \
                                                            feature_y[:,:-1, :, :], flow_backward, metric_y, \
                                                            t, N)
                    mf_gen = GenNet.feature_decode(latent, feature_warped)
                    mf_gen = F.interpolate(mf_gen, (h_f, w_f))
                torch.cuda.synchronize()
                end_time = time.time()
                print("elapse time {}".format(end_time - start_time))
                mf_gen = fromTensor(mf_gen) 
                mf_gen = np.clip(mf_gen, -1.0, 1.0) * 0.5 + 0.5
                writer.append_data(np.uint8(mf_gen * 255.0))
            end_total_time = time.time()
            print("elapse time {}".format(end_total_time - start_total_time))
            if config.verbose:
                toc_gen = time.time()
                LOG("DEBUG")(f"GenNet time elapse {toc_gen - tic_gen}")
    LOG("info")("Test Finish!")
