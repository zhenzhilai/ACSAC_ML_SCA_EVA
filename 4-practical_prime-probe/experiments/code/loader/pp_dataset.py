import os

import torch
from PIL import Image
import numpy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json

class PP_SideChannelDataset_RealTrace_to_IMG(Dataset):
    def __init__(self, data_dir, mode, ID_file, sidechannel_subdir, dim=1, config=None):
        # super(SideChannelDataset, self).__init__()
        image_subdir = 'Celeba_image'
        # sidechannel_subdir = 'simplified_cacheline'
        # sidechannel_subdir = 'Jcacheline_processed'
        # sidechannel_subdir = 'all_idct_prints'
        sidechannel_subdir = sidechannel_subdir

        # current version: len(self.trace_list) = 50000, len(self.image_list) = 162770
        self.image_list = os.listdir(os.path.join(data_dir, image_subdir, mode))
        self.trace_list = os.listdir(os.path.join(data_dir, sidechannel_subdir, mode))
        self.id_file = ID_file
        # Face ID list
        with open(self.id_file, 'r') as f:
            self.ID_dict = json.load(f)
        f.close()
        self.ID_cnt = len(set(self.ID_dict.values()))
        print('Total %d ID.' % self.ID_cnt)

        # order the list
        self.image_list = sorted(self.image_list, key=lambda x: x.split('.jpg')[0])
        self.trace_list = sorted(self.trace_list, key=lambda x: x.split('.npz')[0])
        
        # apply the prefix
        self.image_list = [os.path.join(data_dir, image_subdir, mode, i) for i in self.image_list]
        self.trace_list = [os.path.join(data_dir, sidechannel_subdir, mode, i) for i in self.trace_list]

        if len(self.trace_list) < 20000:
            self.trace_list = self.trace_list[1:]
            self.image_list = self.image_list[1:]
        
        ### Either use 2D encoder or 1D encoder
        self.dim = dim

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.trace_list)

    def __getitem__(self, idx):
        trace_prefix = self.trace_list[idx].split('/')[-1].split('.npz')[0]
        image_prefix = self.image_list[idx].split('/')[-1].split('.jpg')[0]
        assert (trace_prefix == image_prefix), 'error with mismatch. I knew it.'

        trace = numpy.load(self.trace_list[idx])['arr_0']
        trace = torch.tensor(numpy.array(trace), dtype=torch.float)

        # image = numpy.asarray(Image.open(self.image_list[idx]), dtype=numpy.float32)
        image = Image.open(self.image_list[idx])
        image_ID = self.ID_dict[image_prefix+'.jpg']
        
        image = self.transform(image)

        ### image_prefix is used to store images and image_ID is used to train the Classifier.
        if self.dim == 2:
            return trace.unsqueeze(0), image, image_prefix, image_ID
        return trace, image, image_prefix, image_ID
        


class PP_SideChannelDataset_RealTrace_to_IDCT(Dataset):
    def __init__(self, data_dir, mode, ID_file, sidechannel_subdir, dim=1, config=None):
        # super(SideChannelDataset, self).__init__()
        idct_subdir = 'target_activities_data'
        # sidechannel_subdir = 'simplified_cacheline'
        # sidechannel_subdir = 'Jcacheline_processed'
        # sidechannel_subdir = 'all_idct_prints'
        sidechannel_subdir = sidechannel_subdir

        # current version: len(self.trace_list) = 50000, len(self.image_list) = 162770
        self.idct_list = os.listdir(os.path.join(data_dir, idct_subdir, mode))
        self.trace_list = os.listdir(os.path.join(data_dir, sidechannel_subdir, mode))
        self.id_file = ID_file
        # Face ID list
        with open(self.id_file, 'r') as f:
            self.ID_dict = json.load(f)
        f.close()
        self.ID_cnt = len(set(self.ID_dict.values()))
        print('Total %d ID.' % self.ID_cnt)

        # order the list
        self.idct_list = sorted(self.idct_list, key=lambda x: x.split('.npz')[0])
        self.trace_list = sorted(self.trace_list, key=lambda x: x.split('.npz')[0])
        
        # apply the prefix
        self.idct_list = [os.path.join(data_dir, idct_subdir, mode, i) for i in self.idct_list]
        self.trace_list = [os.path.join(data_dir, sidechannel_subdir, mode, i) for i in self.trace_list]

        # if len(self.trace_list) < 20000:
        #     self.trace_list = self.trace_list[1:]
        #     self.idct_list = self.idct_list[1:]
        
        ### Either use 2D encoder or 1D encoder
        self.dim = dim

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.trace_list)

    def __getitem__(self, idx):
        trace_prefix = self.trace_list[idx].split('/')[-1].split('.npz')[0]
        idct_prefix = self.idct_list[idx].split('/')[-1].split('.npz')[0]
        assert (trace_prefix == idct_prefix), 'error with mismatch. I knew it.'

        trace = numpy.load(self.trace_list[idx])['arr_0']
        trace = torch.tensor(numpy.array(trace), dtype=torch.float)

        # image = numpy.asarray(Image.open(self.image_list[idx]), dtype=numpy.float32)
        idct = numpy.load(self.idct_list[idx])['arr_0']
        idct = torch.tensor(numpy.array(idct), dtype=torch.float)
        image_ID = self.ID_dict[idct_prefix+'.jpg']


        ### image_prefix is used to store images and image_ID is used to train the Classifier.
        if self.dim == 2:
            return trace.unsqueeze(0), idct, idct_prefix, image_ID
        return trace, idct, idct_prefix, image_ID
    



class PP_SideChannelDataset_RealTrace_to_IDCT_REDO50w(Dataset):
    def __init__(self, data_dir, mode, ID_file, sidechannel_subdir, dim=1, config=None):
        # super(SideChannelDataset, self).__init__()
        idct_subdir = 'idct_redo50w'
        # sidechannel_subdir = 'simplified_cacheline'
        # sidechannel_subdir = 'Jcacheline_processed'
        # sidechannel_subdir = 'all_idct_prints'
        sidechannel_subdir = sidechannel_subdir

        # current version: len(self.trace_list) = 50000, len(self.image_list) = 162770
        # self.idct_list = os.listdir(os.path.join(data_dir, idct_subdir, mode))
        self.trace_list = os.listdir(os.path.join(data_dir, sidechannel_subdir, mode))
        print("trace_list: ", len(self.trace_list))

        # order the list
        self.trace_list = sorted(self.trace_list, key=lambda x: x.split('.npz')[0])
        
        self.trace_dir = os.path.join(data_dir, sidechannel_subdir, mode)
        self.idct_dir = os.path.join(data_dir, idct_subdir, mode)
        # print("trace_dir: ", self.trace_dir)
        # print("idct_dir: ", self.idct_dir)
        # exit()
        # if len(self.trace_list) < 20000:
        #     self.trace_list = self.trace_list[1:]
        #     self.idct_list = self.idct_list[1:]
        
        ### Either use 2D encoder or 1D encoder
        self.dim = dim

    def __len__(self):
        return len(self.trace_list)

    def __getitem__(self, idx):
        trace_prefix = self.trace_list[idx].split('/')[-1].split('.npz')[0]
        idct_prefix = trace_prefix[:6]
        # print(trace_prefix + " -----" + idct_prefix)
        assert (trace_prefix[:6] == idct_prefix), 'error with mismatch. I knew it.'

        # trace = numpy.load(self.trace_list[idx])['arr_0']
        trace = numpy.load(self.trace_dir + "/" + trace_prefix + '.npz')['arr_0']
        trace = torch.tensor(numpy.array(trace), dtype=torch.float)

        # image = numpy.asarray(Image.open(self.image_list[idx]), dtype=numpy.float32)
        # idct = numpy.load(self.idct_list[idx])['arr_0']
        idct = numpy.load(self.idct_dir + "/" + idct_prefix + '.npz')['arr_0']
        idct = torch.tensor(numpy.array(idct), dtype=torch.float)
        # image_ID = self.ID_dict[idct_prefix+'.jpg']
        image_ID = 0

        # print("trace_prefix: ", trace.shape)
        # print("idct_prefix: ", idct.shape)
        # exit()

        ### image_prefix is used to store images and image_ID is used to train the Classifier.
        if self.dim == 2:
            return trace.unsqueeze(0), idct, idct_prefix, image_ID
        return trace, idct, idct_prefix, image_ID
    

class PP_SideChannelDataset_Recons_IDCT(Dataset):
    def __init__(self, data_dir, mode, ID_file, sidechannel_subdir, dim=1, config=None):
        # super(SideChannelDataset, self).__init__()
        image_subdir = 'Celeba_image'
        # sidechannel_subdir = 'simplified_cacheline'
        # sidechannel_subdir = 'Jcacheline_processed'
        # sidechannel_subdir = 'all_idct_prints'
        sidechannel_subdir = sidechannel_subdir

        # current version: len(self.trace_list) = 50000, len(self.image_list) = 162770
        self.image_list = os.listdir(os.path.join(data_dir, image_subdir, mode))
        self.trace_list = os.listdir(os.path.join(data_dir, sidechannel_subdir, mode))
        self.id_file = ID_file
        # Face ID list
        with open(self.id_file, 'r') as f:
            self.ID_dict = json.load(f)
        f.close()
        self.ID_cnt = len(set(self.ID_dict.values()))
        print('Total %d ID.' % self.ID_cnt)

        # order the list
        self.image_list = sorted(self.image_list, key=lambda x: x.split('.jpg')[0])
        self.trace_list = sorted(self.trace_list, key=lambda x: x.split('.npz')[0])

        # print("trace_list: ", self.trace_list[:10])
        # print("image_list: ", self.image_list[:10])

        # if len(self.image_list) > 50000:
        #     self.image_list = self.image_list[:50000]
        #     self.trace_list = self.trace_list[:50000]
        
        # apply the prefix
        self.image_list = [os.path.join(data_dir, image_subdir, mode, i) for i in self.image_list]
        self.trace_list = [os.path.join(data_dir, sidechannel_subdir, mode, i) for i in self.trace_list]
        
        ### Either use 2D encoder or 1D encoder
        self.dim = dim

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.trace_list)

    def __getitem__(self, idx):
        trace_prefix = self.trace_list[idx].split('/')[-1].split('.npz')[0]
        image_prefix = self.image_list[idx].split('/')[-1].split('.jpg')[0]
        # print(trace_prefix + " -----" + image_prefix)
        # exit()
        assert (trace_prefix == image_prefix), 'error with mismatch. I knew it.'

        trace = numpy.load(self.trace_list[idx])['arr_0']
        trace = torch.tensor(numpy.array(trace), dtype=torch.float)

        # image = numpy.asarray(Image.open(self.image_list[idx]), dtype=numpy.float32)
        image = Image.open(self.image_list[idx])
        image_ID = self.ID_dict[image_prefix+'.jpg'] - 1
        
        image = self.transform(image)

        ### image_prefix is used to store images and image_ID is used to train the Classifier.
        if self.dim == 2:
            return trace.unsqueeze(0), image, image_prefix, image_ID
        return trace, image, image_prefix, image_ID