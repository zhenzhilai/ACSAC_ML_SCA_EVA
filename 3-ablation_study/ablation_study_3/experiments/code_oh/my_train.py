import random
import time
from tqdm import tqdm

import progressbar
import torch
import torch.nn as nn
from pytorch_msssim import ssim
import pytorch_ssim


import models
import utils
from data_loader import *
from loss import SsimLoss
from params import Params

import json
import wandb
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
import argparse



trainset_path = "./data/BB-OH-data/train/"
testset_path = "./data/BB-OH-data/test/"

img_trainset_path = "./data/Celeba_image/train/"
img_testset_path = "./data/Celeba_image/test/"
ID_file = "./data/CelebA_ID_dict.json"

# MODEL_PATH = "./output/ckpts/"
# IMG_OUT_PATH = "./output/"

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Replicate WEBIST-2023-cache",
    #project="Black-box onehot",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-4,
    "architecture": "AE-1D",
    "dataset": "CelebA",
    "epochs": 50,
    },

    # add name to this run
    #name="Black-Box CONV1D AE",
    #name="Coninue q9362fyj from 38",
    #name="Coninue continue from 46",

    save_code=True,

    mode="disabled",
)

TRACE_LEN = 300000

### Test cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
# print(device)

def save_color_img(img, path):

    img = img.clone()
    min = float(img.min())
    max = float(img.max())
    img.clamp_(min = min, max=max)
    img.add_(-min).div_(max-min + 1e-5)
    
    ndarray = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(ndarray).save(path)

class CacheLineDataset(Dataset):
    def __init__(self, trace_dir, image_dir, ID_file):
        super(CacheLineDataset, self).__init__()
        ### either train or test
        self.trace_dir = trace_dir
        self.image_dir = image_dir
        

        self.trace_list = sorted(os.listdir(self.trace_dir))
        self.image_list = sorted(os.listdir(self.image_dir))

        if len(self.trace_list) > 80000:
            self.trace_list = self.trace_list[:80000]
        if len(self.image_list) > 80000:
            self.image_list = self.image_list[:80000]

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),  (0.5, 0.5, 0.5))
        ])

        with open(ID_file, 'r') as f:
            self.ID_dict = json.load(f)

        self.ID_cnt = len(set(self.ID_dict.values()))
        print('Total %d ID.' % self.ID_cnt)

    def __len__(self):
        return len(self.trace_list)

    def __getitem__(self, idx):
        trace_name = self.trace_list[idx]
        trace_prefix = trace_name.split('.')[0]
        image_name = trace_prefix + '.jpg'

        trace = np.load(self.trace_dir + trace_name)
        trace = trace['arr_0']
        trace = torch.tensor(trace, dtype=torch.float)
    
        image = Image.open(self.image_dir + image_name)
        image = self.transform(image)

        ID = self.ID_dict[image_name] - 1
        
        return trace, image, trace_prefix, ID
    

class ImageEngine:
    def __init__(self, args):
        self.args = args
        self.epoch = 0 
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.ssim = SsimLoss()
        self.real_label = 1
        self.fake_label = 0
        self.init_model_optimizer()
        self.train_losses = []
        self.test_losses = []

    def init_model_optimizer(self):
        self.enc = models.TraceEncoder_1DCNN_encode(self.args.MAX_TRACE_LEN, dim=self.args.nz)
        self.enc = self.enc.to(device)

        self.dec = models.__dict__['ResDecoder%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        self.dec = self.dec.to(device)    

        self.optim = torch.optim.Adam(
                        list(self.enc.parameters()) + \
                        list(self.dec.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

        self.E = models.image_output_embed_128(dim=self.args.nz, nc=self.args.nc)
        self.E = self.E.to(device)

        self.D = models.classifier(dim=self.args.nz, n_class=1, use_bn=False)
        self.D = self.D.to(device)

        self.C = models.classifier(dim=self.args.nz, n_class=self.args.num_class, use_bn=False)
        self.C = self.C.to(device)

        self.optim_D = torch.optim.Adam(
                        list(self.E.parameters()) + \
                        list(self.D.parameters()) + \
                        list(self.C.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print(f'Save Model to {path}')
        state = {
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'E': self.E.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print(f'Load Model from {path}')
        ckpt = torch.load(path, map_location=device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])
        self.E.load_state_dict(ckpt['E'])
        self.D.load_state_dict(ckpt['D'])
        self.C.load_state_dict(ckpt['C'])

    def save_state(self, path):
        torch.save({
            'epoch': self.epoch,
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'E': self.E.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict(),
            'optim': self.optim.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'loss': (self.mse, self.l1, self.bce, self.ce, self.ssim),
            'seed': self.args.seed,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
            }, path)

    def load_state(self, path):
        checkpoint = torch.load(path)
        self.enc.load_state_dict(checkpoint['enc'])
        self.dec.load_state_dict(checkpoint['dec'])
        self.E.load_state_dict(checkpoint['E'])
        self.D.load_state_dict(checkpoint['D'])
        self.C.load_state_dict(checkpoint['C'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.optim_D.load_state_dict(checkpoint['optim_D'])
        self.epoch = checkpoint['epoch']
        self.mse, self.l1, self.bce, self.ce, self.ssim = checkpoint['loss']
        self.args.seed = checkpoint['seed']
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        torch.manual_seed(self.args.seed)

    def save_output(self, output, path):
        utils.save_image(output.data, path, normalize=True)

    def zero_grad_G(self):
        self.enc.zero_grad()
        self.dec.zero_grad()
        
    def zero_grad_D(self):
        self.E.zero_grad()
        self.D.zero_grad()
        self.C.zero_grad()

    def set_train(self):
        self.enc.train()
        self.dec.train()
        self.E.train()
        self.D.train()
        self.C.train()

    def set_eval(self):
        self.enc.eval()
        self.dec.eval()
        self.E.eval()
        self.D.eval()
        self.C.eval()

    def _train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record = utils.Record()
            record_mse = utils.Record()
            record_G = utils.Record()
            record_D = utils.Record()
            record_C_real = utils.Record()
            record_C_fake = utils.Record()
            record_C_real_acc = utils.Record()
            record_C_fake_acc = utils.Record()
            start_time = time.time()
            
            
            # progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start() #progressbar.ProgressBar(maxval=len(data_loader)).start()
            # for i, (trace, image, prefix, ID) in enumerate(data_loader):

            tbar = tqdm(range(len(data_loader)), ncols=180)
            data_loader = iter(data_loader)
            for i in tbar:
                trace, image, prefix, ID = next(data_loader)
                # progress.update(i + 1)
                image = image.to(device)
                trace = trace.to(device)
                ID = ID.to(device)
                bs = image.size(0)

                # train D with real
                self.zero_grad_D()
                real_data = image.to(device)
                batch_size = real_data.size(0)
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).to(device)
                label_fake = torch.full((batch_size, 1), self.fake_label, dtype=real_data.dtype).to(device)

                embed_real = self.E(real_data)
                output_real = self.D(embed_real)
                errD_real = self.bce(output_real, label_real)
                D_x = output_real.mean().item()

                # train D with fake
                encoded = self.enc(trace)
                noise = torch.randn(bs, self.args.nz).to(device)
                decoded = self.dec(encoded + 0.05 * noise)
                
                output_fake = self.D(self.E(decoded.detach()))
                errD_fake = self.bce(output_fake, label_fake)
                D_G_z1 = output_fake.mean().item()
                
                errD = errD_real + errD_fake
                
                # train C with real
                pred_real = self.C(embed_real)
                errC_real = self.ce(pred_real, ID)

                (errD_real + errD_fake + errC_real).backward()
                self.optim_D.step()
                record_D.add(errD.item())
                record_C_real.add(errC_real.item())
                record_C_real_acc.add(utils.accuracy(pred_real, ID))

                # train G with D and C
                self.zero_grad_G()

                encoded = self.enc(trace)

                noise = torch.randn(bs, self.args.nz).to(device)
                decoded = self.dec(encoded + 0.05 * noise)

                embed_fake = self.E(decoded)
                output_fake = self.D(embed_fake)
                pred_fake = self.C(embed_fake)

                errG = self.bce(output_fake, label_real)
                errC_fake = self.ce(pred_fake, ID)
                mse_loss = self.mse(decoded, image)
                ssim_loss = self.ssim(decoded, image)
                # recons_err = 0.84 * mse_loss + ssim_loss
                recons_err = self.args.alpha * ssim_loss + (1 - self.args.alpha) * mse_loss

                (errG + errC_fake + self.args.lambd * recons_err).backward()
                D_G_z2 = output_fake.mean().item()
                self.optim.step()
                record_G.add(errG.item())
                record_mse.add(mse_loss.item())
                record.add(recons_err.item())
                record_C_fake.add(errC_fake.item())
                record_C_fake_acc.add(utils.accuracy(pred_fake, ID))

                if i == 0:
                    path_to_train_images = self.args.image_root + "train/"
                    utils.make_path(path_to_train_images)
                    self.save_output(decoded, os.path.join(path_to_train_images, (f'train_{self.epoch:03d}.jpg')))
                    self.save_output(image, os.path.join(path_to_train_images, (f'train_{self.epoch:03d}_target.jpg')))

                wandb.log({
                    "Train MSE Loss": mse_loss.item(),
                    "Recons Loss": recons_err.item(),
                    "Loss of G": errG.item(),
                    "Loss of D": errD.item(),
                    "Loss of C ID real": errC_real.item(),
                    "Loss of C ID fake": errC_fake.item(),
                    "D(x)": D_x,
                    "D(G(z1))": D_G_z1,
                    "D(G(z2))": D_G_z2
                })

                tbar.set_description('Epoch %d, Loss %.4f, Loss_G %.4f, Loss_D %.4f, Loss_C_real %.4f, Loss_C_fake %.4f, Acc_C_real %.4f, Acc_C_fake %.4f' % (self.epoch, record.mean(), record_G.mean(), record_D.mean(), record_C_real.mean(), record_C_fake.mean(), record_C_real_acc.mean(), record_C_fake_acc.mean()))


            # progress.finish()
            # utils.clear_progressbar()
            self.train_losses.append(record.mean())
            tbar.close()    

            # print('----------------------------------------')
            # print(f'Epoch: {self.epoch}')
            # print(f'Costs Time: {(time.time() - start_time):.2f} s')
            # print(f'MSE Loss: {(record_mse.mean()):.6f}')
            # print(f'Recons Loss: {(record.mean()):.6f}')
            # print(f'Loss of G: {(record_G.mean()):.6f}')
            # print(f'Loss of D: {(record_D.mean()):.6f}')
            # print(f'Loss & Acc of C ID real: {(record_C_real.mean()):.6f} & {(record_C_real_acc.mean()):.6f}')
            # print(f'Loss & Acc of C ID fake: {(record_C_fake.mean()):.6f} & {(record_C_fake_acc.mean()):.6f}')
            # print(f'D(x) is: {D_x:.6f}, D(G(z1)) is: {D_G_z1:.6f}, D(G(z2)) is: {D_G_z2:.6f}')

            



    def train(self, train_loader, test_loader):
        test_freq = self.args.test_freq
        exp_name = self.args.exp_name
        output_root = self.args.output_root
        ckpt_root = self.args.ckpt_root
        
        for i in range(self.epoch, self.args.num_epoch):
            wandb.log({"Epoch": i})
            self._train(train_loader)
            if i % test_freq == 0:
                self._test(test_loader)
                self.save_model(os.path.join(ckpt_root, f'{(i+1):03d}.pth'))
            self.save_state(os.path.join(output_root, exp_name, 'temp_state.pth'))
        self.save_model(os.path.join(ckpt_root, 'final.pth'))
        os.remove(os.path.join(output_root, exp_name, 'temp_state.pth'))
        print('Training finished!')
        self.print_min_loss()

    def _test(self, data_loader):
        print("Start Testing...")
        self.set_eval()
        record_mse = utils.Record()
        record = utils.Record()
        start_time = time.time()
        # progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start() #progressbar.ProgressBar(maxval=len(data_loader)).start()
        
        ### create a folder to store the output images
        img_test_path = self.args.image_root + "test/"
        utils.make_path(img_test_path)
        img_test_path = os.path.join(img_test_path, f'{self.epoch:03d}')
        utils.make_path(img_test_path)

        with torch.no_grad():
            # for i, (trace, image, prefix, ID) in enumerate(data_loader):
            ssim_record = torch.zeros(1)
            
            tbar_test = tqdm(range(len(data_loader)), ncols=180)
            data_loader = iter(data_loader)
            for i in tbar_test:
                trace, image, prefix, ID = next(data_loader)
                # progress.update(i + 1)
                image = image.to(device)
                trace = trace.to(device)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)                
                mse_loss = self.mse(decoded, image)
                ssim_loss = self.ssim(decoded, image)

                ssim_value = pytorch_ssim.ssim(decoded, image).detach()
                ssim_record = ssim_record + ssim_value.cpu()
                # recons_err = 0.84 * mse_loss + ssim_loss
                recons_err = self.args.alpha * ssim_loss + (1 - self.args.alpha) * mse_loss
                record_mse.add(mse_loss.item())
                record.add(recons_err.item())

                # if i == 0:
                # print("test: ", i, prefix)
                ### save each decoded image
                # if i == 0:
                for idx in range(len(prefix)):
                    image_output_path = os.path.join(img_test_path, (f'{prefix[idx]}.jpg'))
                    save_color_img(decoded[idx], image_output_path)

                        # image_target_path = os.path.join(img_test_path, (f'{prefix[idx]}_target.jpg'))
                        # save_color_img(image[idx], image_target_path)

                tbar_test.set_description('Epoch %d, Recons Loss %.4f, MSE LOSS %.4f, SSIM %.4f' % (self.epoch, record.mean(), record_mse.mean(), ssim_record.item()/max(i+1, 1)))

                # self.save_output(decoded, os.path.join(img_test_path, (f'{i}.jpg')))
                # self.save_output(image, os.path.join(img_test_path, (f'{i}_target.jpg')))
                if i == 0:
                    if self.epoch == 1:
                        wandb.log({"Target Image: ": [wandb.Image(image[:5], caption="Epoch {}".format(self.epoch))]})
                    wandb.log({"Test Image: ": [wandb.Image(decoded[:5], caption="Epoch {}".format(self.epoch))]})
                    
                
                wandb.log({
                    "Test MSE Loss": record_mse.mean(),
                    "TEST Recons Loss": record.mean(),
                    "TEST SSIM": ssim_record.item()/max(i+1, 1)
                })
            # progress.finish()
            # utils.clear_progressbar()
            self.test_losses.append(record.mean())
            tbar_test.close()
            # print('----------------------------------------')
            # print('Test')
            # print(f'Costs Time: {(time.time() - start_time):.2f} s')
            # print(f'MSE Loss: {(record_mse.mean()):.6f}')
            # print(f'Recons Loss: {(record.mean()):.6f}')
            

    def print_min_loss(self):
        train_np = np.array(self.train_losses)
        test_np = np.array(self.test_losses)
        print(f'Minimum training loss: {(train_np.min().item()):.6f} in epoch {train_np.argmin().item() + 1}')
        print(f'Minimum testing loss: {(test_np.min().item()):.6f} in epoch {test_np.argmin().item() * self.args.test_freq + 1}')

    def save_image(self, output, name_list, path):
        assert len(output) == len(name_list)
        for i in range(len(output)):
            name = name_list[i]
            export_path = os.path.join(path, f'{name}.jpg')
            utils.save_image(output[i].unsqueeze(0).data,
                             export_path,
                             normalize=True, nrow=1, padding=0)
            
    def inference(self, data_loader):
        """
        For the given model,
          1. print the average ssim score
          2. generate reconstruction images
        """
        recons_dir = os.path.join(self.args.output_root, self.args.exp_name, 'recons')
        target_dir = os.path.join(self.args.output_root, self.args.exp_name, 'target')
        utils.make_path(recons_dir)
        utils.make_path(target_dir)

        self.set_eval()
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        with torch.no_grad():
            ssim_sum = 0
            ssim_cnt = 0
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(device)
                trace = trace.to(device)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)
                decoded = decoded.to('cpu')

                ssim_val = ssim(decoded, image, data_range=1, size_average=True)
                batch_size = image.size(0)
                ssim_sum += ssim_val * batch_size
                ssim_cnt += batch_size

                self.save_image(image, prefix, target_dir)
                self.save_image(decoded, prefix, recons_dir)

        progress.finish()
        utils.clear_progressbar()
        print(f'Avg. SSIM score: {ssim_sum / ssim_cnt}')


class My_args:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--exp_name', type=str, required=True)
        parser.add_argument('--MAX_TRACE_LEN', type=int, default=300000)
        parser.add_argument('--num_epoch', type=int, default=50)
        parser.add_argument('--num_workers', type=int, default=3)
        parser.add_argument('--test_freq', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--nz', type=int, default=128)
        parser.add_argument('--nc', type=int, default=3)
        parser.add_argument('--image_size', type=int, default=128)
        parser.add_argument('--num_class', type=int, default=10177)
        parser.add_argument('--lambd', type=float, default=100)
        parser.add_argument('--alpha', type=float, default=0.84)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--image_root', type=str, default="./image_out_default/")
        parser.add_argument('--ckpt_root', type=str, default="./ckpt_out_default/")
        parser.add_argument('--output_root', type=str, default="./output_default/")
        parser.add_argument('--load_ckpt', type=str, default= '')

        
        self.args = parser.parse_args()

    def parse(self):
        return self.args
    




if __name__ == '__main__':
    p = My_args()
    args = p.parse()

    print(args.exp_name)    

    args.image_root = os.path.join("./output/", args.exp_name+"/", "images/")
    args.ckpt_root = os.path.join("./output/", args.exp_name+"/", 'ckpt')
    

    engine = ImageEngine(args)
    

    utils.make_path(args.output_root)
    utils.make_path(os.path.join(args.output_root, args.exp_name))
    utils.make_path("./output/")
    utils.make_path("./output/"+args.exp_name)
    utils.make_path(args.image_root)
    utils.make_path(args.ckpt_root)



    print(args.image_root)
    print(args.ckpt_root)
    args.seed = 666 #random.randint(1, 10000)

    wandb.config.update(args)

    torch.manual_seed(args.seed)
    loader = DataLoader(args)

    train_dataset = CacheLineDataset(trainset_path, img_trainset_path, ID_file)
    test_dataset  = CacheLineDataset(testset_path, img_testset_path, ID_file)
    train_loader  = loader.get_loader(train_dataset)
    test_loader  = loader.get_loader(test_dataset, shuffle=False)

    if args.load_ckpt != '':
        print("load ckpt from: " + args.load_ckpt)
        engine.load_model(args.load_ckpt)
        engine._test(test_loader)
    else:
        engine.train(train_loader, test_loader)

