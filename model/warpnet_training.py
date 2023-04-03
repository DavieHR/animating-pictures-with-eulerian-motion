import torch
import numpy as np
from tqdm import tqdm
from model.coach import Coach
from util.helper import Glint_log as LOG
from util.flow_utils import flow2img, flow2diff
from util.euler_integration import compute_meshgrid
from util.softsplat import ModuleSoftsplat
from math import ceil

splatting_op = ModuleSoftsplat("average")

class WarpNetTrainer(Coach):
    def __init__(self, opt):
        super(WarpNetTrainer, self).__init__(opt)
        self.network_names = {"Gen": opt.Gen_name, "Dis": opt.Dis_name}
        self.use_estimation_net = False
        self.joint_training = opt.joint_training
        self.no_target_norm = opt.no_target_norm
        self.norm_flow = opt.norm_flow
        merge_parameter = None
        if opt.EMotion_name is not None:
            self.network_names["EMotion"] = opt.EMotion_name
            self.use_estimation_net = True
            merge_parameter = "Gen+EMotion"
        if opt.distributed:
            self.type_par = "DDP"
        self.config = self.load_config()
        self.net = self.register_network()
        self.loss = self.register_loss()
        self.optimizers = self.register_optimizer(merge_parameter)
        self.schedulers = self.register_scheduler()
        self.print_networks(True)
        
    def train(self, epoch):
        
        LOG("info")("Training Start -----> epoch [{}]".format(epoch))

        #pbar = tqdm(range(len(self.train_data)), initial = 0, dynamic_ncols = True, smoothing = 0.01)

        for k, v in self.net.items():
            self.net[k] = v.train()

        for idx, data in enumerate(self.train_data):
            self.total_steps = epoch * len(self.train_data) + idx
            ff, lf, mf, flow, t, N, (h_ori, w_ori) = data
            h,w = ff.shape[2:]
            ff = ff.to(self.device)
            mf = mf.to(self.device)
            lf = lf.to(self.device)
            if self.use_estimation_net:
                flow_gt = flow.to(self.device)
                if not self.no_target_norm:
                    grid_source = compute_meshgrid(flow_gt.shape).to(self.device)
                    flow_gt = 2 * (flow_gt + grid_source) - 1
                n,c,h,w = ff.shape
                h_ori = h_ori.view(n,1,1,1)
                w_ori = w_ori.view(n,1,1,1)
                divisor = 2 ** 4
                w_, h_ = int(ceil(w / divisor) * divisor),int(ceil(h / divisor) * divisor)
                if h_ != h or w_ != w:
                    ff = F.interpolate(ff, (h_, w_))
                flow = self.net['EMotion'](ff)
                #flow = flow * 16
                if h_ != h or w_ != w:
                    flow = F.interpolate(flow, (h, w))
                if self.norm_flow:
                    flow[:,0:1,:,:] = flow[:,0:1,:,:] * w_ori.view(-1,1,1,1)
                    flow[:,1:2,:,:] = flow[:,1:2,:,:] * h_ori.view(-1,1,1,1)
            else:
                flow = flow.to(self.device)

            if self.joint_training:
                mf_gen, flow_forward, flow_backward, metric_x, metric_y  = self.net['Gen'](ff, lf, flow, t, N)
            else:
                mf_gen, flow_forward, flow_backward, metric_x, metric_y  = self.net['Gen'](ff, lf, flow.detach(), t, N)
            
            ### Discriminator Loss.
            self.net['Dis'].module.enable_backward()
            for _ in range(self.opt.d_num):
                self.optimizers['Dis'].zero_grad()
                loss = torch.zeros((1)).to(mf_gen)
                fake_labels,_ = self.net['Dis'](mf_gen.detach())
                real_labels,_ = self.net['Dis'](mf)

                if not isinstance(fake_labels, list):
                    fake_labels = [fake_labels]
                    real_labels = [real_labels]
                for (fake_label, real_label) in zip(fake_labels, real_labels):
                    loss += self.loss['StyleGanLoss']['loss'](fake_label, real_label)
                self.visor["Text" + "Discriminator"] = loss.item()
                loss.backward()
                self.optimizers['Dis'].step()

            self.net['Dis'].module.disable_backward()
            total_loss = torch.zeros((1)).to(self.device)
            self.optimizers['Gen'].zero_grad()
            ### Generator Loss
            for k, v in self.loss.items():
                if k == 'StyleGanLoss':
                    fake_labels, features_fake = self.net['Dis'](mf_gen, True)
                    real_labels, features_real = self.net['Dis'](mf, True)
                    if not isinstance(fake_labels, list):
                        fake_labels = [fake_labels]
                        real_labels = [real_labels]
                        features_fake = [features_fake]
                        features_real = [features_real]

                    loss_g = torch.zeros((1)).to(mf_gen)
                    loss_g_feature = torch.zeros((1)).to(mf_gen)
                    for idx, fake_label in enumerate(fake_labels):
                        loss_g_per_item, loss_g_feature_per_item = v['loss'](fake_label, None, features_fake[idx], features_real[idx])
                        loss_g += loss_g_per_item
                        loss_g_feature += loss_g_feature_per_item
                    loss = v['weight'] * loss_g + v['other']['feature_weight'] * loss_g_feature
                    self.visor["Text" + k] = loss_g.item()
                    self.visor["Text" + k + "_feature_matching"] = loss_g_feature.item()
                elif 'flow' in k:
                    loss = v['weight'] * v['loss'](flow, flow_gt, ff, lf, h_ori, w_ori)
                    self.visor["Text" + k] = loss.item()
                else:
                    loss = v['weight'] * v['loss'](mf_gen, mf)
                    self.visor["Text" + k] = loss.item()
                total_loss += loss
            total_loss.backward()
            self.optimizers['Gen'].step()

            if self.total_steps % self.opt.vis_interval == 0:
                self.visor["ImageGen"] = 0.5 * mf_gen + 0.5
                self.visor["ImageFF"] = 0.5 * ff + 0.5
                self.visor["ImageLF"] = 0.5 * lf + 0.5
                self.visor["ImageGT"] = 0.5 * mf + 0.5

                self.visor["Imagemetric_ff"] = metric_x.exp() / metric_x.exp().max()
                self.visor["Imagemetric_lf"] = metric_y.exp() / metric_y.exp().max()

                SplatWarp_forward = splatting_op(ff, flow_forward, torch.ones((1,1,20,20)))
                SplatWarp_backward = splatting_op(lf, flow_backward, torch.ones((1,1,20,20)))

                self.visor["ImageSplatWarp_forward"] = SplatWarp_forward * 0.5 + 0.5
                self.visor["ImageSplatWarp_backward"] = 0.5 * SplatWarp_backward + 0.5

                flow_n_to_show = torch.zeros_like(ff)
                n = flow_forward.size(0)
                for i in range(n):
                    flow_to_show = flow_forward[i].detach().cpu().squeeze().permute((1,2,0)).numpy()
                    flow_to_show = flow2img(flow_to_show) / 255.0
                    flow_to_show = torch.from_numpy(flow_to_show.astype(np.float32)).permute((2,0,1)).unsqueeze(0)
                    flow_n_to_show[i,...] = flow_to_show

                self.visor["Imageflow_forward"] = flow_n_to_show
                n = flow_backward.size(0)
                flow_back_n_to_show = torch.zeros_like(ff)
                for i in range(n):
                    flow_to_show = flow_backward[i].detach().cpu().squeeze().permute((1,2,0)).numpy()
                    flow_to_show = flow2img(flow_to_show) / 255.0
                    flow_to_show = torch.from_numpy(flow_to_show.astype(np.float32)).permute((2,0,1)).unsqueeze(0)
                    flow_back_n_to_show[i,...] = flow_to_show

                self.visor["Imageflow_backward"] = flow_back_n_to_show
                if self.use_estimation_net:
                    flow_n_to_show = torch.zeros_like(ff)
                    if not self.no_target_norm:
                        flow_detach = (flow.detach() + 1) * 0.5 - grid_source
                        flow_detach_gt = (flow_gt.detach() + 1) * 0.5 - grid_source
                    else:
                        flow_detach = flow.detach()
                        flow_detach_gt = flow_gt.detach()
                    self.visor["HistGen"] = flow_detach
                    self.visor["HistGT"] = flow_detach_gt
                    n = flow.size(0)
                    for i in range(n):
                        flow_to_show = flow_detach[i].cpu().squeeze().permute((1,2,0)).numpy()
                        flow_to_show = flow2img(flow_to_show) / 255.0
                        flow_to_show = torch.from_numpy(flow_to_show.astype(np.float32)).permute((2,0,1)).unsqueeze(0)
                        flow_n_to_show[i,...] = flow_to_show
                    self.visor["Imageflow"] = flow_n_to_show
                    flow_gt_n_to_show = torch.zeros_like(ff)
                    for i in range(n):
                        flow_to_show = flow_detach_gt[i].detach().cpu().squeeze().permute((1,2,0)).numpy()
                        flow_to_show = flow2img(flow_to_show) / 255.0
                        flow_to_show = torch.from_numpy(flow_to_show.astype(np.float32)).permute((2,0,1)).unsqueeze(0)
                        flow_gt_n_to_show[i,...] = flow_to_show
                    self.visor["Imageflow_gt"] = flow_gt_n_to_show
                    flow_diff = flow2diff(mf, flow.detach())
                    self.visor["Imageflow_diff"] = flow_diff
                self.visor(self.total_steps, False)

        if epoch % self.opt.save_interval == 0:
            #self.eval(epoch)
            self.save_networks(epoch)

        self.update_optimizers(epoch)
        if self.opt.re_init_dataset:
            self.train_data = data_func("train", opt)
            self.test_data = data_func("test", opt)
            LOG("info")("re_initialize dataset, current data size is {}".format(len(self.train_data)))

    def eval(self, epoch):
        #pbar = tqdm(range(len(self.test_data)), initial = 0, dynamic_ncols = True, smoothing = 0.01)
        self.net['Gen'].eval()
        if self.use_estimation_net:
            self.net['EMotion'].eval()
        for idx, data in enumerate(self.test_data):
            if self.use_estimation_net:
                ff = data[0]
                ff = ff.to(self.device)
                flow = self.net['EMotion'](ff)
            else:
                ff, flow = data
                ff = ff.to(self.device)
                flow = flow.to(self.device)
            N,C,H,W = ff.shape
            mf_gen_in_time = ff.unsqueeze(0)
            for t in range(1,31):
                t = torch.ones((1)) * t                
                N = torch.ones((1)) * 30.0
                with torch.no_grad():
                    mf_gen  = self.net['Gen'](ff, ff, flow, t.to(ff).int(), N.to(ff))
                mf_gen = torch.clamp(mf_gen, -1.0, 1.0)
                mf_gen_in_time = torch.cat((mf_gen_in_time, mf_gen.unsqueeze(0)), dim = 0)
            self.visor["VideoToShow"] = mf_gen_in_time
            self.visor(epoch)

    def test(self):
        import os
        import cv2
        import numpy as np
        import imageio
        from util.flow_utils import flow2img
        self.net['Gen'].eval()
        if self.use_estimation_net:
            self.net['EMotion'].eval()
        for idx, data in enumerate(self.test_data):
            if self.use_estimation_net:
                ff = data[0]
                ff = ff.to(self.device)
                flow = self.net['EMotion'](ff)
            else:
                ff, flow = data
                ff = ff.to(self.device)
                flow = flow.to(self.device)
            N,C,H,W = ff.shape
            mf_gen_in_time = (ff.squeeze().detach().permute((1,2,0)).cpu().numpy()) * 0.5 + 0.5
            show_vec = [mf_gen_in_time]

            show_flow = []
            for t in range(31):
                t = torch.ones((1)) * t                
                N = torch.ones((1)) * 30.0
                with torch.no_grad():
                    mf_gen = self.net['Gen'](ff, ff, flow, t.to(ff).int(), N.to(ff))
                mf_gen = mf_gen.squeeze().detach().cpu().permute((1,2,0)).numpy()

                mf_gen = np.clip(mf_gen, -1.0, 1.0)
                show_vec.append(mf_gen * 0.5 + 0.5)
            if not os.path.exists(self.opt.save_path):
                os.makedirs(self.opt.save_path)
            imageio.mimsave(os.path.join(self.opt.save_path, f'{idx}_' + self.opt.save_file_name), [np.uint8(x * 255.0) for x in show_vec], fps = 10.0)

    
