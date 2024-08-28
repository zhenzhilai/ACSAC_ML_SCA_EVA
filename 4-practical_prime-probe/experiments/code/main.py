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
from utils.utils import load_checkpoint, load_checkpoint_for_test

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="ACSAC_2024_practical2image",

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
    name="Test practical traces to images",
    
    mode="disabled",

    # save_code=True,

    # notes="1D-encoder and 2D-decoder -> (384 * 64) * 64; maxpool, use stride instead, 24576 --> 3072 "
)


def main(hyp_var):
    seed = 666 #random.randint(1, 10000)


    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # for singe GPU training only
    network = MyModel(hyp_var)
    network = network.cuda()

    # optimiser = torch.optim.Adam(network.parameters(), lr=hyp_var.lr, betas=hyp_var.betas,
    #                              weight_decay=hyp_var.weight_decay)

    # classifier = network.classifier
    # embedder = network.embed

    ### optimiser for both classifier and embedder
    # optimiser_classifier = torch.optim.Adam(list(classifier.parameters()) + list(embedder.parameters()), lr=hyp_var.lr, betas=hyp_var.betas, weight_decay=hyp_var.weight_decay)

    # optimiser_classifier = torch.optim.Adam(classifier.parameters(), lr=hyp_var.lr, betas=hyp_var.betas, weight_decay=hyp_var.weight_decay)
    
    # embedder = network.embed

    # train_set = SideChannelDataset(hyp_var.data_dir, mode='train', ID_file=hyp_var.ID_file)
    
    # ### print the shape of the first data
    # print(train_set[0][0].shape, train_set[0][1].shape, train_set[0][2], train_set[0][3])
    # exit(1)
    
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
        if not hyp_var.recon_trace_to_img:
            train_loader = SideChannel(pp_dataset.PP_SideChannelDataset_RealTrace_to_IDCT(hyp_var.data_dir, mode='train', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim), config=hyp_var)
            test_loader = SideChannel(pp_dataset.PP_SideChannelDataset_RealTrace_to_IDCT(hyp_var.data_dir, mode='test', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim),
                                # update drop_last and shuffle
                                config=EasyDict({**hyp_var, **{'drop_last': False, 'shuffle': False}}))
            hyp_var['iters_per_epoch'] = len(train_loader)
            
    # elif hyp_var.pp_to_img:
    #     train_loader = SideChannel(pp_dataset.PP_SideChannelDataset_RealTrace_to_IMG(hyp_var.data_dir, mode='train', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim), config=hyp_var)
    #     test_loader = SideChannel(pp_dataset.PP_SideChannelDataset_RealTrace_to_IMG(hyp_var.data_dir, mode='test', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim),
    #                             # update drop_last and shuffle
    #                             config=EasyDict({**hyp_var, **{'drop_last': False, 'shuffle': False}}))
    else:
        if not hyp_var.recon_trace_to_img:
            train_loader = SideChannel(SideChannelDataset(hyp_var.data_dir, mode='train', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim), config=hyp_var)
            test_loader = SideChannel(SideChannelDataset(hyp_var.data_dir, mode='test', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim),
                                # update drop_last and shuffle
                                config=EasyDict({**hyp_var, **{'drop_last': False, 'shuffle': False}}))
            hyp_var['iters_per_epoch'] = len(train_loader)
            

    # trainer = Trainer(loss=torch.nn.functional.l1_loss, optimiser=optimiser, param=hyp_var)
    # trainer = Trainer(loss=pytorch_ssim.ssim, optimiser=optimiser, param=hyp_var)
    trainer = Trainer(param=hyp_var)
    validator = Validator(hyp_var)


    # for idx, (x, y) in enumerate(dataloader):
    if hyp_var.recon_trace_to_img:
        pretrained_model = hyp_var.load_ckpt
        # trace_to_idct_to_img_model_dir = "./ckpts/simplified_ideal_trace_to_img/"
        # for ep in range(hyp_var.epochs):
        # ep = hyp_var.R_input_recons_activity_ep
        test_loader = SideChannel(pp_dataset.PP_SideChannelDataset_Recons_IDCT(hyp_var.data_dir, mode='test', ID_file=hyp_var.ID_file, sidechannel_subdir= hyp_var.subdir, dim=hyp_var.dim),
                            # update drop_last and shuffle
                            config=EasyDict({**hyp_var, **{'drop_last': False, 'shuffle': False}}))
        ckpt_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained_model, map_location=ckpt_device)
        network.load_state_dict(checkpoint['model'])
        network.eval()
        validator.valid(-2, data_loader=test_loader, model=network, img_out_path=hyp_var.img_out_path)
        return False

    print(hyp_var.target)

    # if hyp_var.test_from_ckpt != "-1":
    #     print("Load checkpoint for testing........")
    #     network = load_checkpoint_for_test(network, hyp_var.test_from_ckpt)
    #     network.eval()
    #     if hyp_var.target == 'idct':
    #         validator.valid_Trace2IDCT_01(epoch=-10, data_loader=test_loader, model=network, img_out_path=hyp_var.img_out_path)
    #     else:
    #         validator.valid(epoch=-10, data_loader=test_loader, model=network, img_out_path=hyp_var.img_out_path)
    #     return False

    if args.load_ckpt != '':
        checkpoint_path = args.load_ckpt
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        ckpt_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=ckpt_device)
        network.load_state_dict(checkpoint['model'])
        network.eval()
        if hyp_var.target == 'idct':
            validator.valid_Trace2IDCT_01(-1, test_loader, model=network, img_out_path=hyp_var.img_out_path)
        else:
            validator.valid(-1, test_loader, model=network, img_out_path=hyp_var.img_out_path)
        return False

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
    
    parser.add_argument('--idct_out_dim', default=384, type=int,
                        help='output idct trace num')
        
    parser.add_argument('--recon_trace_to_img', default=0, type=int,
                        help='Use 1 for load pre_trained model and convert 01 trace to images')

    parser.add_argument('--R_ep', default=0, type=int,
                        help='choose the epoch for the Reconstructor loaded in reconstructed to img')
    
    parser.add_argument('--R_input_recons_activity_ep', default=0, type=int,
                        help='choose the epoch for recons input to Reconstructor loaded in reconstructed to img')
    
    parser.add_argument('--test_from_ckpt', default="-1", type=str,
                        help='load the checkpoint for testing')
    
    parser.add_argument('--load_ckpt', default='', type=str,
                        help='load ckpt from path')
    args = parser.parse_args()
    args.ckpts_dir = f"./ckpts/{args.experiment_name}"
    args.img_out_path = f"./image_output/{args.experiment_name}"
    args.trace_lens = args.trace_lens
    args.dim = args.dim

    from configs.config import C
    main(EasyDict({**C, **vars(args)}))

