import os
import torch
from tqdm import tqdm
# from pytorch_msssim import ssim
import pytorch_ssim
from utils.utils import save_image, save_color_img
import wandb

from sklearn.metrics import accuracy_score, classification_report
import numpy as np


class Validator:
    def __init__(self, configs):
        super(Validator, self).__init__()
        self.param = configs
        upload_train_image_flag = True
    

    def valid(self, epoch, data_loader, model, img_out_path):
        print('start validation:')
        model.eval()
        store_figure_test_path = os.path.join(img_out_path, str(epoch))
        ### make folder
        if not os.path.exists(store_figure_test_path):
            os.makedirs(store_figure_test_path)
        if not os.path.exists(os.path.join(store_figure_test_path, 'test')):
            os.makedirs(os.path.join(store_figure_test_path, 'test'))
        if not os.path.exists(os.path.join(img_out_path, 'target')):
            os.makedirs(os.path.join(img_out_path, 'target'))

        tbar = tqdm(range(len(data_loader)), ncols=135)
        data_loader = iter(data_loader)
        ssim_record = torch.zeros(1)
        # for batch_idx, (trace, image) in enumerate(data_loader):
        for batch_idx in tbar:
            trace, image, prefix, ID = next(data_loader)    
            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            logits = model(trace)

            ssim_value = pytorch_ssim.ssim(logits.cuda(), image, window_size=8).detach()
            ssim_record = ssim_record + ssim_value.cpu()
            tbar.set_description('epoch {} metrics {}'.format(epoch,  ssim_record.item()/(batch_idx+1)))
            
            wandb.log({"Test ssim": ssim_value.cpu()})

            ### upload image to wandb
            if batch_idx == 0:
                if epoch == 0:
                    wandb.log({"Target image": [wandb.Image(image[:5], caption="Epoch {}".format(epoch))]})   
                    self.save_output(image, os.path.join(img_out_path, (f'test_{epoch:05d}_target.jpg')))

                wandb.log({"Test image": [wandb.Image(logits[:5], caption="Epoch {}".format(epoch))]})
                self.save_output(logits, os.path.join(img_out_path, (f'test_{epoch:05d}.jpg')))

            
            ### save each image
            for tmp_idx in range(len(image)):
                # save_color_img(logits[tmp_idx], os.path.join(store_figure_test_path, 'test', (f'{tmp_idx:05d}.jpg')))
                save_color_img(logits[tmp_idx], os.path.join(store_figure_test_path, 'test', (f'{prefix[tmp_idx]}.jpg')))

            del image, trace
        self.save_ckpts(model, epoch)

    def save_ckpts(self, model, epoch):
        path = os.path.join(self.param.ckpts_dir, str(epoch)+'.pth')
        if not os.path.exists(self.param.ckpts_dir):
            os.makedirs(self.param.ckpts_dir)
        print(f'save model to {path}')
        state = {
            'model': model.state_dict(),
            'epoch': epoch,
            'param': self.param,
        }
        torch.save(state, path)

    def save_output(self, output, path):
        ### if path does not exist, make path
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        save_image(output.data, path, normalize=True)
        # save_color_img(output, path)



    def valid_Trace2IDCT_01(self, epoch, data_loader, model, img_out_path):
        print('start validation:')

        store_figure_test_path = os.path.join(img_out_path, str(epoch))
        ### make folder
        if not os.path.exists(store_figure_test_path):
            os.makedirs(store_figure_test_path)
        if not os.path.exists(os.path.join(store_figure_test_path, 'test')):
            os.makedirs(os.path.join(store_figure_test_path, 'test'))
        if not os.path.exists(os.path.join(img_out_path, 'target')):
            os.makedirs(os.path.join(img_out_path, 'target'))

        tbar = tqdm(range(len(data_loader)), ncols=135)
        data_loader = iter(data_loader)
        # ssim_record = torch.zeros(1)
        acc_record = []
        # for batch_idx, (trace, image) in enumerate(data_loader):
        for batch_idx in tbar:
            # print("enter")
            trace, image, prefix, ID = next(data_loader)    
            # print("trace", trace.shape)

            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            logits = model(trace)



            shapes = logits.shape
            # print("shapes", shapes)
            # exit()
            pred_out = torch.round(logits)

            pred_out = pred_out.cpu().detach().numpy().astype(int).tolist()
            image = image.cpu().detach().numpy().astype(int).tolist()

            pred_labels = np.array(pred_out).reshape([-1, 64])
            true_labels = np.array(image).reshape([-1, 64])

            # print("pred_labels", pred_labels.shape)
            # print("true_labels", true_labels.shape)

            # print("pred_labels", pred_labels[0])
            # print("true_labels", true_labels[0])
            # exit()

            report_accuracy = accuracy_score(true_labels, pred_labels)
            # print(report_accuracy)
            # print("Accuracy is ", report_accuracy)
            acc_record.append(report_accuracy)

            # ### save as csv
            # for i in range(len(pred_out)):
            #     # pred_path = IMG_OUT_PATH + data_flag + str(ep+1) + '/' + in_filenames[i][:-4] + 'pred.csv'
            #     pred_path = store_figure_test_path + "/test/" + prefix[i][:6] + '.csv'
            #     f_pre = open(pred_path, 'w')
            #     f_pre.close()
            #     f_pre = open(pred_path, 'a')
            #     for line in pred_out[i]:
            #         f_pre.write(' '.join(map(lambda x:str(x), line)) + '\n')
            #     f_pre.close()

            ### save as npz
            for i in range(len(pred_out)):
                pred_path = store_figure_test_path + "/test/" + prefix[i][:6] + '.npz'
                np.savez_compressed(pred_path, arr_0=pred_out[i])

            # ssim_value = pytorch_ssim.ssim(logits.cuda(), image, window_size=8).detach()
            # ssim_record = ssim_record + ssim_value.cpu()
            tbar.set_description('epoch {} metrics {}'.format(epoch,  report_accuracy))
            
            wandb.log({"Test Acc": report_accuracy})

            # ### upload image to wandb
            # if batch_idx == 0:
            #     if epoch == 0:
            #         wandb.log({"Target image": [wandb.Image(image[:5], caption="Epoch {}".format(epoch))]})   
            #         self.save_output(image, os.path.join(img_out_path, (f'test_{epoch:05d}_target.jpg')))

            #     wandb.log({"Test image": [wandb.Image(logits[:5], caption="Epoch {}".format(epoch))]})
            #     self.save_output(logits, os.path.join(img_out_path, (f'test_{epoch:05d}.jpg')))

            
            # ### save each image
            # for tmp_idx in range(len(image)):
                # if batch_idx == 0:
                #     # save_color_img(image[tmp_idx], os.path.join(img_out_path, 'target', (f'{tmp_idx:05d}_target.jpg')))
                #     save_color_img(image[tmp_idx], os.path.join(img_out_path, 'target',  (f'{prefix[tmp_idx]}.jpg')))
                
                # save_color_img(logits[tmp_idx], os.path.join(store_figure_test_path, 'test', (f'{tmp_idx:05d}.jpg')))
                # save_color_img(logits[tmp_idx], os.path.join(store_figure_test_path, 'test', (f'{prefix[tmp_idx]}.jpg')))

            # del image, trace

        print("**** Accuracy is ", np.mean(acc_record))
        wandb.log({"Epoch Average Test Acc": np.mean(acc_record)})
        self.save_ckpts(model, epoch)
