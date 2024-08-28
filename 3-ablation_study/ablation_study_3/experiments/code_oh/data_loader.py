import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import utils
from PIL import Image
from torch.utils.data import Dataset
from utils import Printer

def side_to_bound(side):
    if side == 'cacheline32':
        v_max = 0xFFFF_FFFF >> 6
        v_min = -(0xFFFF_FFFF >> 6)
    elif side == 'cacheline':
        v_max = 0xFFF >> 6
        v_min = -(0xFFF >> 6)
    else:
        raise NotImplementedError
    return v_max, v_min

class CelebaDataset(Dataset):
    def __init__(self, npz_dir, img_dir, ID_path, split,
                image_size, side, trace_c, trace_w,
                trace_len, leng=None, attack=None, img_type=None):
        super().__init__()
        self.npz_dir = os.path.join(npz_dir, split)
        self.img_dir = os.path.join(img_dir, split)
        self.trace_c = trace_c
        self.trace_w = trace_w
        self.trace_len = trace_len
        self.attack = attack
        self.img_type = img_type

        print(leng)
        exit(1)
        self.npz_list = sorted(os.listdir(self.npz_dir))[:leng]
        self.img_list = sorted(os.listdir(self.img_dir))[:leng]

        self.transform = transforms.Compose([
                       transforms.Resize(image_size),
                       transforms.CenterCrop(image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])

        # For Manifold
        # self.v_max, self.v_min = side_to_bound(side)
        
        Printer.print(f'{split.capitalize()} set: {len(self.npz_list)} Data Points.')

        with open(ID_path, 'r') as f:
            self.ID_dict = json.load(f)

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        prefix = npz_name.split('.')[0]
        suffix = '.jpg' if self.img_type == 'jpg' else '.webp'
        img_name = prefix + suffix
        ID = int(self.ID_dict[prefix + '.jpg']) - 1

        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']

        # For Manifold
        # trace = np.pad(trace, (0, 93216), mode='constant')  # Pad 256*256*6 - 300,000 = 93216 zeros
        # trace = trace.astype(np.float32)
        # trace = torch.from_numpy(trace)
        # trace = trace.view([self.trace_c, self.trace_w, self.trace_w])
        # trace = utils.my_scale(v=trace, v_max=self.v_max, v_min=self.v_min)

        # For proposed
        trace = self.full_to_cacheline_index_encode(trace, self.trace_len)
        trace = torch.from_numpy(trace)

        image = Image.open(os.path.join(self.img_dir, img_name))
        image = self.transform(image)

        ID = torch.LongTensor([ID]).squeeze()
        return trace, image, prefix, ID
    
    def to_cacheline(self, addr):
        if self.img_type == 'jpg':
            return (abs(addr) & 0xFFF) >> 6
        elif self.img_type == 'webp':
            return abs(addr) >> 6
        else:
            raise NotImplementedError

    def full_to_cacheline_index_encode(self, full: np.array, trace_len: int):
        assert full.shape[0] <= trace_len, "Error: trace length longer than padding length"
        arr = np.zeros((trace_len, 64), dtype=np.float16)
        arr_cacheline = self.to_cacheline(full)

        if self.attack == 'pp':
            arr[np.arange(len(arr_cacheline)), arr_cacheline] = 1
        elif self.attack == 'wb':
            result = np.where(full > 0, 1., -1.)
            arr[np.arange(len(arr_cacheline)), arr_cacheline] = result
        else:
            raise NotImplementedError

        return arr.astype(np.float32)

class ImageDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.img_dir = os.path.join(args.image_dir, split)

        self.img_list = sorted(os.listdir(self.img_dir))

        self.transform = transforms.Compose([
                       transforms.Resize(args.image_size),
                       transforms.CenterCrop(args.image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])
        
        print(f'{split.capitalize()} set: {len(self.img_list)} Data Points.')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        image = Image.open(os.path.join(self.img_dir, img_name))
        image = self.transform(image)
        
        return image

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.init_param()

    def init_param(self):
        self.gpus = torch.cuda.device_count()
        self.gpus = max(1, self.gpus)

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers= self.args.num_workers,
                            shuffle=shuffle
                        )
        return data_loader
