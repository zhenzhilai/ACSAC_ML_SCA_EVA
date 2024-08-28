import os
import json
import torch
import argparse

class Params:
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Training Options
        parser.add_argument('--exp_name', type=str, required=True)
        parser.add_argument('--output_root', type=str, default='../output/')
        parser.add_argument('--dataset', type=str, choices=['CelebA_jpg', 'CelebA_webp'])
        parser.add_argument('--num_epoch', type=int, default=50)
        parser.add_argument('--num_workers', type=int, default=3)
        parser.add_argument('--side', type=str, choices=['cacheline32', 'cacheline', 'cacheline_encode'], default="cacheline")
        parser.add_argument('--attack', type=str, choices=['pp', 'wb'], help='"pp" for Prome+Probe attack and "wb" for write-back channel attack')
        parser.add_argument('--test_freq', type=int, default=1)
        parser.add_argument('--use_refiner', type=int, default=0, choices=[0, 1])

        # Hyper Parameters
        parser.add_argument('--trace_c', type=int, default=-1)
        parser.add_argument('--trace_w', type=int, default=-1)
        parser.add_argument('--batch_size', type=int, default=2) ## 50
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--nz', type=int, default=512)
        parser.add_argument('--nc', type=int, default=3)
        parser.add_argument('--noise_pp_op', type=str, default='', choices=['', 'out', 'flip', 'order'])
        parser.add_argument('--noise_pp_k', type=float, default=0)
        parser.add_argument('--lambd', type=float, default=100)
        parser.add_argument('--alpha', type=float, default=0.84)
        parser.add_argument('--dropout', type=float, default=0.5)

        # Pre-train VAE specific parameters
        parser.add_argument('--vae_epoch', type=int, default=100)
        parser.add_argument('--kld_weight', type=float, default=0.00025)
        parser.add_argument('--vae_lr', type=float, default=1e-4)

        self.args = parser.parse_args()
        if torch.cuda.is_available():
            self.args.device = torch.device('cuda')
        else:
            print("Warning: CUDA is not available, the calculation will be unreasonably slow")
            self.args.device = torch.device('cpu')
        self.parser = parser
        self.get_data_path('dataset_info.json')

    def parse(self):
        return self.args

    def get_data_path(self, json_path):
        with open(json_path, 'r') as f:
            data_path = json.load(f)
        root_dir = '../'
        for dataset_name in data_path.keys():
            item = data_path[dataset_name]
            for k in item.keys():
                if k in ['img_dir', 'pin', 'cacheline32', 'cacheline', 'cacheline_encode', 'pp-intel-dcache', 'pp-intel-icache', 'ID_path', 'refiner_path']:
                    item[k] = os.path.join(root_dir, item[k])
        self.args.data_path = data_path
        print('data_path: ', data_path)
        self.args.image_size = data_path[self.args.dataset]['image_size']

    def save_params(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.args.__dict__, f)

    def print_options(self, params):
        message = '----------------- Params ---------------\n'
        for k, v in sorted(vars(params).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

if __name__ == '__main__':
    p = Params()
    args = p.parse()
    p.print_options(args)
