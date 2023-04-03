import os
import os.path
import random
import cv2
from functools import partial
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.flo'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dirs):
    images = []

    dir_mapping = {}
    assert os.path.isdir(dirs), '%s is not a valid directory' % dirs
    for root, _, fnames in sorted(os.walk(dirs)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                dir_path = os.path.dirname(path).split('/')[-1]
                if dir_path in dir_mapping:
                    dir_mapping[dir_path] += [path]
                else:
                    dir_mapping[dir_path] = [path]
    for k, v in dir_mapping.items():
        images.append(v)
    return images

def make_dataset_list(dirs):
    images = []
    assert os.path.isdir(dirs), '%s is not a valid directory' % dirs

    for root, _, fnames in sorted(os.walk(dirs)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def readFlow(fn, norm = False):
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
            data = np.resize(data, (int(h), int(w), 2))
            if norm:
                data[..., 0] = data[..., 0] / w
                data[..., 1] = data[..., 1] / h

            return data

class VideoFlowData:
    def __init__(self, 
                 data_root,
                 data_type, 
                 opt,
                ):

        self.data_type = data_type
        if not opt.sep_data:
            data_path = os.path.join(data_root, data_type)
        else:
            data_path = data_root
        self.data = make_dataset(data_path)
        self.resolution_size = opt.crop_size
        self.flow_file_name = opt.flow_file_name
        self.norm_flow = opt.norm_flow
        self.no_flow_require = opt.EMotion_name is not None
        self.mask_dir = opt.mask_dir
        if opt.neg_data_path is not None:
            self.neg_data = make_dataset_list(opt.neg_data_path)
        else:
            self.neg_data = None
        self.neg_prob = opt.neg_prob

    def aug_data(self, datasets):
        function_compose = []

        def value_flip(x, direction):
            if x.shape[2] == 2:
                for _d in direction:
                    x[..., _d - 1] *= -1
                return x
            return x

        if random.random() < 0.5:
            # flip random.
            flip_dims = [[1],[2],[1,2]]
            direction = random.choice(flip_dims)
            function_compose.append(partial(torch.flip, dims = direction))
            function_compose.append(partial(value_flip, direction = direction))
        for i in range(len(datasets)):
            _data = datasets[i]
            h, w = _data.shape[:2]
            if h*w < 1280*720:
                if h < w:
                    target_size = (1280, 720)
                else:
                    target_size = (720, 1280)
                datasets[i] = cv2.resize(_data, target_size)
    
        h,w = datasets[0].shape[:2]
        if h <= self.resolution_size:
            h = self.resolution_size + 1
        if w <= self.resolution_size:
            w = self.resolution_size + 1
        x_select = list(range(self.resolution_size // 2, w - self.resolution_size + self.resolution_size // 2))
        y_select = list(range(self.resolution_size // 2, h - self.resolution_size + self.resolution_size // 2))
        random_center_x = random.choice(x_select)
        random_center_y = random.choice(y_select)
        x_left, x_right = random_center_x - self.resolution_size // 2, random_center_x + self.resolution_size - self.resolution_size // 2
        y_left, y_right = random_center_y - self.resolution_size // 2, random_center_y + self.resolution_size - self.resolution_size // 2
    
        def _unzip_compose_function(x, i):
            for _f in x:
                i = _f(i)
            return i

        for idx, data in enumerate(datasets):
            if self.data_type == 'train':
                _data_in = data[y_left:y_right, x_left: x_right, :].copy()
                _h_crop, _w_crop = _data_in.shape[:2]
                if _h_crop != self.resolution_size or _w_crop != self.resolution_size:
                    _data_in = cv2.resize(_data_in, (self.resolution_size, self.resolution_size))
            else:
                _data_in = data.copy()
            data_trans = self._transform_to_data(_data_in)
            if self.data_type == 'train':
                data_trans = _unzip_compose_function(function_compose, data_trans)
            datasets[idx] = data_trans
        return datasets
                
    def _transform_to_data(self, x):
        x = torch.from_numpy(x)
        x = x.permute((2,0,1))
        x = x.to(torch.float32)
        return x

    def __getitem__(self, idx):
        datas = self.data[idx]
        
        video_datas = list(filter(lambda x: x.split('.')[0].split('_')[-1] == 'video', datas))
        if len(video_datas) == 0:
            common_file_name = datas[0].split('/')[-2]
            video_datas = []
            potential_list = [str(x) for x in list(range(0, 60))]
            for data in datas:
                if data.split('.')[0].split('_')[-1] in potential_list:
                    video_datas.append(data)
            pos_index = -1
            if 'norm' in self.flow_file_name:
                self.flow_file_name = common_file_name + "_motion_norm.flo"
            else:
                self.flow_file_name = common_file_name + "_motion.flo"
        else:
            common_file_name = datas[0].split('/')[-2].replace(''.join(['_'] + datas[0].split('/')[-2].split('_')[-1:]),'')[6:]
            pos_index = 0
        video_datas = sorted(video_datas, key=lambda d:int(d.split('/')[-1].split('.')[0].split('_')[pos_index]))
        common_path = os.path.dirname(video_datas[-1])
        if self.data_type == "train":
            if self.neg_data is not None:
                prob_neg = random.random()
            else:
                prob_neg = 1.0
            if prob_neg < self.neg_prob:
                index_random = random.choice(list(range(len(self.neg_data))))
                video_start = cv2.imread(self.neg_data[index_random])[...,::-1]
                video_mid = video_start.copy()
                video_end = video_start.copy()
                h,w = video_start.shape[:2]
                flow_start = np.zeros((h, w, 2))
                random_start_idx = random_mid_idx = 0
                random_end_idx = random_mid_idx + 1
            else:
                _length_data = list(range(len(video_datas)))
                random_start_idx = random.choice(_length_data)
                random_end_idx = random.choice(list(range(len(video_datas) - random_start_idx))) + random_start_idx
                random_mid_idx = random.choice(list(range(random_start_idx, random_end_idx + 1)))
                video_start = cv2.imread(video_datas[random_start_idx])[...,::-1]
                video_end = cv2.imread(video_datas[random_end_idx])[...,::-1]
                video_mid = cv2.imread(video_datas[random_mid_idx])[...,::-1]
                flow_average_path = os.path.join(common_path, self.flow_file_name)
                flow_start = readFlow(flow_average_path, self.norm_flow)
                def _check_mask(x, mask_dir_list):
                    for mask_dir in mask_dir_list:
                        mask_path = os.path.join(mask_dir, x) + '_mask.png'
                        if os.path.exists(mask_path):
                            return mask_path
                    return None
                if self.mask_dir is not None:
                    mask_path = _check_mask(common_file_name, self.mask_dir)
                    if mask_path is not None:
                        mask = cv2.imread(mask_path,0)[...,np.newaxis] / 255.0
                        flow_start = flow_start * mask
            
            h_ori, w_ori = video_start.shape[:2]
            data_flow = [video_start, video_end, video_mid, flow_start]
        elif self.data_type == "test":
            video_mid = cv2.imread(video_datas[-1], -1)[...,::-1]
            if not self.no_flow_require:
                flow_datas = list(filter(lambda x: x.split('.')[0].split('_')[-1] == 'flow', datas))
                flow_datas = sorted(flow_datas, key=lambda d:int(d.split('/')[-1].split('_')[0]))
                flow_start = readFlow(flow_datas[-1])
                data_flow = [video_mid, flow_start]
            else:
                data_flow = [video_mid]
        datasets = self.aug_data(data_flow)
        yield_data = datasets[:-1] if len(datasets) != 1 else datasets
        for index, x in enumerate(yield_data):
            x = x / 255.0
            x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
            datasets[index] = x
        if self.data_type == "train":
            datasets.append(random_mid_idx - random_start_idx)
            datasets.append(random_end_idx - random_start_idx + 1)
            datasets.append((h_ori, w_ori))
        return datasets

    def __len__(self):
        return len(self.data)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def VideoFlow(data_type, opt):

    if data_type == "train":
        Dataset = []
        for data_path in opt.data_root:
            Dataset += [VideoFlowData(data_path, data_type, opt)]
        Dataset = torch.utils.data.ConcatDataset(Dataset)
    else:
        Dataset = VideoFlowData(opt.data_root[0],data_type, opt)

    batchSize = opt.batchSize if data_type == "train" else 1
    if data_type == "train":
        batchSize *= opt.gpu_nums
    
    return DataLoader(Dataset, batch_size=batchSize, sampler=data_sampler(Dataset, data_type == "train", opt.distributed), num_workers=int(opt.data_worker_num), drop_last=data_type == "train")



