import os
import numpy
from easydict import EasyDict

C = EasyDict()

C.seed = 666

C.notes = "xxxx"
""" root directory """
C.repo_name = 'side_channel'
C.root_dir = os.path.realpath("../")
#C.root_dir = os.path.realpath("../")
C.subdir = 'all_idct_prints_original' # 'simplified_cacheline'

# modify the data_dir
C.data_dir = os.path.join(C.root_dir, 'data')
C.ID_file = os.path.join(C.data_dir, 'CelebA_ID_dict.json')

""" dataset setup """
C.augment = False
C.shuffle = True
C.num_workers = 15
C.drop_last = True
C.dim = 1 #### 2D or 1D encoder

""" model setup """
C.trace_lens = 24576 #24576 #### AVX full size is 300000, our simplified verion is 384
C.latent_dims = 128

""" optimiser """
C.betas = [0.9, 0.999]
C.weight_decay = 0.0001 #0.001

""" training setup """
C.lr = 1e-3
C.batch_size = 8
C.lr_power = 0.9
C.epochs = 100 #50 #60

""" wandb setup """
# Specify you wandb environment KEY; and paste here
C.wandb_key = ""

# Your project [work_space] name
C.project_name = C.repo_name + ''

C.experiment_name = "baseline-test"

# False for debug; True for visualize
C.wandb_online = False

""" save setup """
C.path_to_saved_dir = "./ckpts/exp"
# C.path_to_cache_dir = ".cache/"
C.ckpts_dir = os.path.join(C.path_to_saved_dir, C.experiment_name)
# C.cache_dir = os.path.join(C.path_to_saved_dir, C.experiment_name)

""" model output"""
C.img_out_path = os.path.join(C.root_dir, 'image_output', C.experiment_name)

""" test setup """


import pathlib
pathlib.Path(C.ckpts_dir).mkdir(parents=True, exist_ok=True)

