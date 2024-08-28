import torch
import argparse
from easydict import EasyDict
from engine.train import Trainer
from engine.valid import Validator
from loader.dataset import SideChannelDataset
from loader.dataloader import SideChannel
from model.model import MyModel
# import pp codes
import loader.pp_dataset as pp_dataset
# import pytorch_ssim
from loss.loss import SCA_Loss
from utils.utils import load_checkpoint

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="ACSAC 2024 ultra mse loss",
    # project="ACSAC_2024_newloss",

    # track hyperparameters and run metadata
    # config={
    # "learning_rate": 2e-4,
    # "batch_size": 100,
    # "architecture": "1D-encoder and 2D-decode",
    # "dataset": "CelebA, 384 * 64 * 64",
    # "epochs": 25,
    # "loss": "0.14* l1 + 0.86 * (1-ms-ssim)", ### https://arxiv.org/pdf/1511.08861.pdf
    # "pool": "maxpool",
    # },

    # add name to this run
    # name="new encoder-decoder model testing",
    name="Ablation study 1 2",
    
    mode="disabled",

    save_code=True,

    # notes="1D-encoder and 2D-decoder -> (384 * 64) * 64; maxpool, use stride instead, 24576 --> 3072 "
)


def main(hyp_var):
    seed = 666 #random.randint(1, 10000)


    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # for single GPU training only
    network = MyModel(hyp_var)
    network = network.cuda()
    
    ### set wandb configures
    wandb.config.update(hyp_var)
    wandb.run.name = hyp_var.experiment_name
    ### update wandb notes
    wandb.notes = hyp_var.notes
    # wandb.project = 'paper_experiments_encoder_decoder'


    print("exp name", hyp_var.experiment_name)
    print("sub_dir is ", hyp_var.subdir)
    print("ckpt at: ", hyp_var.ckpts_dir)
    print("traces len is ", hyp_var.trace_lens)
    print("notes:", hyp_var.notes)
    print("Encoder: {}dim encoder".format(hyp_var.dim))

    if hyp_var.target == 'idct':
        train_loader = SideChannel(pp_dataset.PP_SideChannelDataset_RealTrace_to_IDCT(hyp_var.data_dir, mode='train', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim), config=hyp_var)
        test_loader = SideChannel(pp_dataset.PP_SideChannelDataset_RealTrace_to_IDCT(hyp_var.data_dir, mode='test', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim),
                                # update drop_last and shuffle
                                config=EasyDict({**hyp_var, **{'drop_last': False, 'shuffle': False}}))
    else:
        train_loader = SideChannel(SideChannelDataset(hyp_var.data_dir, mode='train', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim), config=hyp_var)
        test_loader = SideChannel(SideChannelDataset(hyp_var.data_dir, mode='test', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim),
                                # update drop_last and shuffle
                                config=EasyDict({**hyp_var, **{'drop_last': False, 'shuffle': False}}))

    hyp_var['iters_per_epoch'] = len(train_loader)
    trainer = Trainer(param=hyp_var)
    validator = Validator(hyp_var)

    ### Load the checkpoint
    if hyp_var.load_ckpt != '':
        print("=> loading checkpoint '{}'".format(hyp_var.load_ckpt))
        ckpt_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(hyp_var.load_ckpt, map_location=ckpt_device)
        network.load_state_dict(checkpoint['model'])
        network.eval()
        
        if hyp_var.target == 'idct':
            validator.valid_Trace2IDCT_01(epoch, test_loader, model=network, img_out_path=hyp_var.img_out_path)
        else:
            validator.valid(0, test_loader, model=network, img_out_path=hyp_var.img_out_path)
        
        return False

    print(hyp_var.target)
    for epoch in range(0, hyp_var.epochs):
        wandb.log({"epoch": epoch})
        if hyp_var.target == 'idct':
            network.train()
            trainer.train_idct_new(epoch, train_loader, model=network)
            network.eval()
            validator.valid_Trace2IDCT_01(epoch, test_loader, model=network, img_out_path=hyp_var.img_out_path)
        else:
            network.train()
            if hyp_var.GAN == 'FULL':
                print("full gan")
                trainer.train_with_D_C(epoch, train_loader, model=network)
            elif hyp_var.GAN == 'D':
                trainer.train_with_D(epoch, train_loader, model=network)
            elif hyp_var.GAN == 'C':
                trainer.train_with_C(epoch, train_loader, model=network)
            elif hyp_var.GAN == 'MANIFOLD':
                trainer.train_with_MANIFOLD(epoch, train_loader, model=network)
            else:
                print("normal train")
                trainer.train(epoch, train_loader, model=network)
            # trainer.train_GAN(epoch, train_loader, model=network, classifier=classifier, embedder=embedder, optimiser_classifier=optimiser_classifier)
            network.eval()
            validator.valid(epoch, test_loader, model=network, img_out_path=hyp_var.img_out_path)
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For paper experiments')

    
    # training hyp-value settings
    parser.add_argument('--batch_size', default=2, type=int)

    parser.add_argument('--epochs', default=50, type=int,
                        help="total epochs that used for the training")

    parser.add_argument('--lr', default=2e-4, type=float,
                        help='default learning rate')
    
    parser.add_argument('--subdir', default='all_idct_prints_original', type=str,
                        help='subdir for the sidechannel data')
    parser.add_argument('--experiment_name', default='baseline-test', type=str,
                        help='experiment name')
    parser.add_argument('--trace_lens', default=384, type=int,
                        help='traces length')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='number of workers for the dataloader')

    parser.add_argument('--notes', default='default notes', type=str,
                        help='notes for the experiment')
    
    parser.add_argument('--dim', default=1, type=int,
                        help='Use 2 for 2D encoder; 1 for 1D encoder')
    
    parser.add_argument('--attn', default=0, type=int,
                        help='Use 1 for ATTN; 0 for no ATTN')
    
    parser.add_argument('--GAN', default='NONE', type=str,
                        help='Use D for CLF_D; C for CLF_C; FULL for both')
    
    parser.add_argument('--target', default="img", type=str,
                        help='output img or idct')
    
    parser.add_argument('--pp_trace_to_img', default=0, type=int,
                        help='Use 1 for load pre_trained model and convert 01 trace to images')
                        
    parser.add_argument('--load_ckpt', default='', type=str,
                        help='load_ckpt from path')

    args = parser.parse_args()
    args.img_out_path = f"./image_output/{args.experiment_name}"
    args.trace_lens = args.trace_lens
    args.dim = args.dim

    from configs.config import C
    main(EasyDict({**C, **vars(args)}))

