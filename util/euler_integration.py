import torch 
import torch.nn.functional as F


def compute_meshgrid(shape, norm = True):
    N, C, H, W = shape
    if norm:
        rows = torch.arange(0, H, dtype=torch.float32) / (H)
        cols = torch.arange(0, W, dtype=torch.float32) / (W)
    else:
        rows = torch.arange(0, H, dtype=torch.float32)
        cols = torch.arange(0, W, dtype=torch.float32)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid[::-1]).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid

def get_flow(flow, 
             cordinate):
    h,w = flow.shape[2:]
    cordinate[:,0,...] = cordinate[:,0,...] / w
    cordinate[:,1,...] = cordinate[:,1,...] / h
    cordinate = (cordinate - 0.5) * 2
    return F.grid_sample(flow, cordinate.permute((0,2,3,1)), align_corners=True, padding_mode="reflection")

def clip_flow(flow, cordinate):
    h,w = flow.shape[2:]
    grid = cordinate + flow
    grid[:,0,:,:] = torch.clip(grid[:,0,:,:], 0, w - 1)
    grid[:,1,:,:] = torch.clip(grid[:,1,:,:], 0, h - 1)
    return grid - cordinate
    
def euler_integration(flow, times):
    shape = flow.shape
    cordinate = compute_meshgrid(shape, False).to(flow)
    n, h_flow, w_flow = flow.size(0),flow.size(2), flow.size(3)
    target_flow = torch.zeros_like(flow)
    #norm_factor = torch.cat((torch.ones((n,1,h_flow,w_flow)) * w_flow, torch.ones(n,1,h_flow,w_flow) * h_flow), dim = 1).to(flow)
    #flow = flow / norm_factor
    flow = clip_flow(flow, cordinate)
    for _ in range(times.int()):
        cordinate_cur = target_flow + cordinate
        target_flow = target_flow + get_flow(flow, cordinate_cur)
        target_flow = clip_flow(target_flow, cordinate)
    return target_flow
        
def euler_integration_once(flow, target_flow):
    shape = flow.shape
    cordinate = compute_meshgrid(shape, False).to(flow)
    n, h_flow, w_flow = flow.size(0),flow.size(2), flow.size(3)
    flow = clip_flow(flow, cordinate)
    #norm_factor = torch.cat((torch.ones((n,1,h_flow,w_flow)) * w_flow, torch.ones(n,1,h_flow,w_flow) * h_flow), dim = 1).to(flow)
    #flow = flow / norm_factor
    cordinate_cur = target_flow + cordinate
    target_flow = target_flow + get_flow(flow, cordinate_cur)
    target_flow = clip_flow(target_flow, cordinate)
    return target_flow

class EulerIntegration:
    def __init__(self, 
                 N,
                 use_cache = False,
                 fast_version = False
                ):
        self.N = N
        self.use_cache = use_cache
        self.fast_version = fast_version
        if use_cache:
            self.cache_queue = None


    def storage_cache_queue(self, flow):

        self.cache_queue = {
                                "forward": [flow],
                                "backward": [flow],
                           }
        shape = flow.shape
        cordinate = compute_meshgrid(shape).to(flow)
        h_flow, w_flow = flow.size(2), flow.size(3)
        target_flow_forward = torch.zeros_like(flow)
        target_flow_backward = torch.zeros_like(flow)
        for i in range(self.N):
            cordinate_cur = target_flow_forward + cordinate
            target_flow_forward = target_flow_forward + get_flow(flow, cordinate_cur)
            self.cache_queue["forward"] += [target_flow_forward]

            cordinate_cur = target_flow_backward + cordinate
            target_flow_backward = target_flow_backward + get_flow(-1 * flow, cordinate_cur)
            self.cache_queue["backward"] += [target_flow_backward]

    def __call__(self, flow, t):

        if self.use_cache:
            if self.cache_queue is None:
                self.storage_cache_queue(flow)
            return self.cache_queue["forward"][t], self.cache_queue["backward"][self.N - t]

        return euler_integration(flow, t), euler_integration(-flow, N - t)











