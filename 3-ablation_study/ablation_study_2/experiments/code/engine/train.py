import os
import torch
from tqdm import tqdm
import pytorch_ssim
import wandb
import torch.nn as nn


class Trainer:
    def __init__(self, param):
        super(Trainer, self).__init__()
        
        self.param = param

    # for poly lr decay
    def poly_lr_decay(self, curr_idx):
        return self.param.lr * (1 - curr_idx / (self.param.iters_per_epoch * self.param.epochs)) ** 0.9

    # for step lr decay
    def step_lr_decay(self, curr_idx):
        total_idx = self.param.epochs * self.param.iters_per_epoch
        if total_idx / 3 > curr_idx >= 0: return self.param.lr
        elif 2 * total_idx / 3 >= curr_idx >= total_idx / 3: return self.param.lr / 10
        else: return self.param.lr / 100

    def train(self, epoch, data_loader, model):
        
        tbar = tqdm(range(len(data_loader)), ncols=200)
        data_loader = iter(data_loader)
        
        ssim_record = torch.zeros(1)
        # for batch_idx, (trace, image) in enumerate(data_loader):
        for batch_idx in tbar:
            trace, image, prefix, ID = next(data_loader)    
            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)

            model.zero_grad_encdec()

            logits = model(trace)
           
            train_loss = model.loss_fn.loss_function(logits, image)
            train_loss_mse = model.mse(logits, image)
            (train_loss_mse).backward()
            model.optimiser.step()
            
            #### Just for the record, it does not affect the training process
            ssim_value  = pytorch_ssim.ssim(logits.cuda(), image, window_size=8).detach()
            wandb.log({"Train loss": train_loss.item(), "ssim in train": ssim_value.cpu()})
            ssim_record = ssim_record + ssim_value.cpu()

            # curr_lr = self.poly_lr_decay(curr_idx=self.param.iters_per_epoch * epoch + batch_idx)
            # for i in range(len(model.optimiser.param_groups)):
            #     model.optimiser.param_groups[i]['lr'] = curr_lr
            tbar.set_description('idx {}/{} loss {} metrics {}'.format(batch_idx, self.param.iters_per_epoch,
                                                                       train_loss.item(), ssim_record.numpy()[0]/(batch_idx+1)))

        return
    
    def train_GAN(self, epoch, data_loader, model, classifier, embedder, optimiser_classifier):
        
        tbar = tqdm(range(len(data_loader)), ncols=200)
        data_loader = iter(data_loader)
        self.real_label = 1
        self.fake_label = 0
        self.bce = nn.BCELoss()

        
        ssim_record = torch.zeros(1)
        errD_record = torch.zeros(1)

        for batch_idx in tbar:
            trace, image, prefix, ID = next(data_loader)    
            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            

            ### Train the Discriminator
            classifier.zero_grad()
            embedder.zero_grad()

            real_data = image.cuda()
            batch_size = real_data.size(0)
            label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).cuda()
            label_fake = torch.full((batch_size, 1), self.fake_label, dtype=real_data.dtype).cuda()
            
            embed_real = embedder(real_data)
            output_real = classifier(embed_real)
            errD_real = self.bce(output_real, label_real)
            D_x = output_real.mean().item()
            
            ### Train D with fake
            fake_data = model(trace)
            embed_fake = embedder(fake_data)
            output_fake = classifier(embed_fake.detach())
            errD_fake = self.bce(output_fake, label_fake)

            errD = errD_real + errD_fake
            errD.backward()
            optimiser_classifier.step() ##### Update this, include embed adam. (to change optimizer in main.py)
            errD_record = errD_record + errD.item()

            #### Train the Generator
            logits = model(trace)
            embed_fake = embedder(logits)
            output_fake = classifier(embed_fake)
            errG = self.bce(output_fake, label_real)
           
            train_loss = self.loss.loss_function(logits, image)
            self.optimiser.zero_grad()
            recons_loss = train_loss + errG
            
            recons_loss.backward()
            self.optimiser.step()

            
            #### Just for the record, it does not affect the training process
            ssim_value  = pytorch_ssim.ssim(logits.cuda(), image, window_size=8).detach()
            wandb.log({"Train loss": train_loss.item(), "ssim in train": ssim_value.cpu()})
            wandb.log({"D_real loss:": errD_real.item(), "D_fake loss": errD_fake.item(), "G loss": errG.item()})
            # wandb.log({"D loss": errD.item(), "G loss": errG.item()})
            ssim_record = ssim_record + ssim_value.cpu()

            curr_lr = self.poly_lr_decay(curr_idx=self.param.iters_per_epoch * epoch + batch_idx)
            for i in range(len(self.optimiser.param_groups)):
                self.optimiser.param_groups[i]['lr'] = curr_lr
            tbar.set_description('idx {}/{} loss {} D_r loss {} D_f loss {} G_loss {} metrics {}'.format(batch_idx, self.param.iters_per_epoch,
                                                                       train_loss.item(), errD_real.item(), errD_fake.item(), errG.item(), ssim_record.numpy()[0]/(batch_idx+1)))

        return
    
    def train_with_D_C(self, epoch, data_loader, model):
        print('start training with D_C:')
        tbar = tqdm(range(len(data_loader)), ncols=200)
        data_loader = iter(data_loader)
        self.real_label = 1
        self.fake_label = 0
        
        ssim_record = torch.zeros(1)
        # errD_record = torch.zeros(1)

        for batch_idx in tbar:
            trace, image, prefix, ID = next(data_loader)    
            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            ID = ID.cuda(non_blocking=True)

            #####################################################
            # Train the Discriminator
            model.zero_grad_classifier()
            batch_size = image.size(0)
            label_real = torch.full((batch_size, 1), self.real_label, dtype=image.dtype).cuda()
            label_fake = torch.full((batch_size, 1), self.fake_label, dtype=image.dtype).cuda()

            # D_TF real
            embed_real = model.embed(image)
            d_tf_real = model.CLF_D(embed_real)
            loss_D_TF_real = model.bce(d_tf_real, label_real)

            # D_TF fake
            fake_imgs = model.decoder(model.encoder(trace))
            
            embed_fake = model.embed(fake_imgs.detach())
            d_tf_fake = model.CLF_D(embed_fake)
            loss_D_TF_fake = model.bce(d_tf_fake, label_fake)

            # loss_D_TF = loss_D_TF_real + loss_D_TF_fake

            # D_ID real
            pred_ids_real = model.CLF_C(embed_real)
            loss_D_ID_real = model.ce(pred_ids_real, ID)

            # backward
            loss_D = (loss_D_TF_real + loss_D_TF_fake + loss_D_ID_real).cuda()
            loss_D.backward()
            model.optimiser_classifier.step()

            #####################################################
            # Train the Generator
            # train G
            model.zero_grad_encdec()
            
            pred_imgs = model.decoder(model.encoder(trace))
            embed_pred = model.embed(pred_imgs)
            d_tf = model.CLF_D(embed_pred)
            d_id = model.CLF_C(embed_pred)

            loss_G_img = model.loss_fn.loss_function(pred_imgs, image)
            loss_G_img_mse = model.mse(pred_imgs, image)
            loss_G_tf = model.bce(d_tf, label_real)
            loss_G_id = model.ce(d_id, ID)

            # backward
            loss_G = (100 * (loss_G_img_mse) + loss_G_tf + loss_G_id).cuda()
            loss_G.backward()
            model.optimiser.step()

            # update lr
            # curr_lr = self.poly_lr_decay(curr_idx=self.param.iters_per_epoch * epoch + batch_idx)
            # for i in range(len(model.optimiser.param_groups)):
            #     model.optimiser.param_groups[i]['lr'] = curr_lr
            #     model.optimiser_classifier.param_groups[i]['lr'] = curr_lr

            #### Just for the record, it does not affect the training process
            ssim_value  = pytorch_ssim.ssim(pred_imgs.cuda(), image, window_size=8).detach()
            wandb.log({"Train G: Total loss": loss_G.item(), 
                       "Train G: SCA_Loss": loss_G_img.item(), 
                       "Train G: D": loss_G_tf.item(), 
                       "Train G: C": loss_G_id.item(), 
                       "ssim in train": ssim_value.cpu()})
            wandb.log({"Train CLF: Total loss:": loss_D.item(), 
                       "Train CLF: D_real loss:": loss_D_TF_real.item(), 
                       "Train CLF: D_fake loss:": loss_D_TF_fake.item(), 
                       "Train CLF: C loss:": loss_D_ID_real.item()})
            # wandb.log({"D loss": errD.item(), "G loss": errG.item()})
            ssim_record = ssim_record + ssim_value.cpu()
            tbar.set_description('idx {}/{} G_Total loss {} G_D loss {} G_C loss {} CLF loss {} metrics {}'.format(batch_idx, self.param.iters_per_epoch,
                                                                       loss_G.item(), loss_G_tf.item(), loss_G_id.item(), loss_D.item(), ssim_record.numpy()[0]/(batch_idx+1)))

    def train_with_MANIFOLD(self, epoch, data_loader, model):
        print('start training with manifold:')
        tbar = tqdm(range(len(data_loader)), ncols=200)
        data_loader = iter(data_loader)
        self.real_label = 1
        self.fake_label = 0
        
        ssim_record = torch.zeros(1)
        # errD_record = torch.zeros(1)

        for batch_idx in tbar:
            trace, image, prefix, ID = next(data_loader)    
            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            ID = ID.cuda(non_blocking=True)

            #####################################################
            # Train the Discriminator
            model.zero_grad_classifier()
            batch_size = image.size(0)
            label_real = torch.full((batch_size, 1), self.real_label, dtype=image.dtype).cuda()
            label_fake = torch.full((batch_size, 1), self.fake_label, dtype=image.dtype).cuda()

            # D_TF real
            embed_real = model.embed(image)
            d_tf_real = model.CLF_D(embed_real)
            loss_D_TF_real = model.bce(d_tf_real, label_real)

            # D_TF fake
            fake_imgs = model.decoder(model.encoder(trace))
            
            embed_fake = model.embed(fake_imgs.detach())
            d_tf_fake = model.CLF_D(embed_fake)
            loss_D_TF_fake = model.bce(d_tf_fake, label_fake)

            # loss_D_TF = loss_D_TF_real + loss_D_TF_fake

            # D_ID real
            pred_ids_real = model.CLF_C(embed_real)
            loss_D_ID_real = model.ce(pred_ids_real, ID)

            # backward
            loss_D = (loss_D_TF_real + loss_D_TF_fake + loss_D_ID_real).cuda()
            loss_D.backward()
            model.optimiser_classifier.step()

            #####################################################
            # Train the Generator
            # train G
            model.zero_grad_encdec()
            
            pred_imgs = model.decoder(model.encoder(trace))
            embed_pred = model.embed(pred_imgs)
            d_tf = model.CLF_D(embed_pred)
            d_id = model.CLF_C(embed_pred)

            loss_G_img = model.loss_fn.loss_function(pred_imgs, image)
            loss_G_img_mse = model.mse(pred_imgs, image)
            loss_G_tf = model.bce(d_tf, label_real)
            loss_G_id = model.ce(d_id, ID)

            # backward
            #loss_G = (100 * (0.84 *loss_G_img + 0.16 * loss_G_img_mse) + loss_G_tf + loss_G_id).cuda()
            loss_G = (100 * (loss_G_img_mse) + loss_G_tf + loss_G_id).cuda()
            loss_G.backward()
            model.optimiser.step()

            # update lr
            #curr_lr = self.poly_lr_decay(curr_idx=self.param.iters_per_epoch * epoch + batch_idx)
            #for i in range(len(model.optimiser.param_groups)):
                #model.optimiser.param_groups[i]['lr'] = curr_lr
                #model.optimiser_classifier.param_groups[i]['lr'] = curr_lr

            #### Just for the record, it does not affect the training process
            ssim_value  = pytorch_ssim.ssim(pred_imgs.cuda(), image, window_size=8).detach()
            wandb.log({"Train G: Total loss": loss_G.item(), 
                       "Train G: SCA_Loss": loss_G_img.item(), 
                       "Train G: D": loss_G_tf.item(), 
                       "Train G: C": loss_G_id.item(), 
                       "ssim in train": ssim_value.cpu()})
            wandb.log({"Train CLF: Total loss:": loss_D.item(), 
                       "Train CLF: D_real loss:": loss_D_TF_real.item(), 
                       "Train CLF: D_fake loss:": loss_D_TF_fake.item(), 
                       "Train CLF: C loss:": loss_D_ID_real.item()})
            # wandb.log({"D loss": errD.item(), "G loss": errG.item()})
            ssim_record = ssim_record + ssim_value.cpu()
            tbar.set_description('idx {}/{} G_Total loss {} G_D loss {} G_C loss {} CLF loss {} metrics {}'.format(batch_idx, self.param.iters_per_epoch,
                                                                       loss_G.item(), loss_G_tf.item(), loss_G_id.item(), loss_D.item(), ssim_record.numpy()[0]/(batch_idx+1)))



    def train_with_D(self, epoch, data_loader, model):
        tbar = tqdm(range(len(data_loader)), ncols=200)
        data_loader = iter(data_loader)
        self.real_label = 1
        self.fake_label = 0
        
        ssim_record = torch.zeros(1)
        # errD_record = torch.zeros(1)

        for batch_idx in tbar:
            trace, image, prefix, ID = next(data_loader)    
            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            ID = ID.cuda(non_blocking=True)

            #####################################################
            # Train the Discriminator
            model.zero_grad_classifier_D()
            batch_size = image.size(0)
            label_real = torch.full((batch_size, 1), self.real_label, dtype=image.dtype).cuda()
            label_fake = torch.full((batch_size, 1), self.fake_label, dtype=image.dtype).cuda()

            # D_TF real
            embed_real = model.embed(image)
            d_tf_real = model.CLF_D(embed_real)
            loss_D_TF_real = model.bce(d_tf_real, label_real)

            # D_TF fake
            fake_imgs = model.decoder(model.encoder(trace))
            
            embed_fake = model.embed(fake_imgs.detach())
            d_tf_fake = model.CLF_D(embed_fake)
            loss_D_TF_fake = model.bce(d_tf_fake, label_fake)

            # backward
            loss_D = (loss_D_TF_real + loss_D_TF_fake).cuda()
            loss_D.backward()
            model.optimiser_classifier_D.step()

            #####################################################
            # Train the Generator
            # train G
            model.zero_grad_encdec()
            
            pred_imgs = model.decoder(model.encoder(trace))
            embed_pred = model.embed(pred_imgs)
            d_tf = model.CLF_D(embed_pred)

            loss_G_img = model.loss_fn.loss_function(pred_imgs, image)
            loss_G_img_mse = model.mse(pred_imgs, image)
            loss_G_tf = model.bce(d_tf, label_real)

            # backward
            # loss_G = (100 * (0.84 * loss_G_img + 0.16 * loss_G_img_mse) + loss_G_tf).cuda()
            loss_G = (100 * (loss_G_img_mse) + loss_G_tf).cuda()
            loss_G.backward()
            model.optimiser.step()

            # # update lr
            # curr_lr = self.poly_lr_decay(curr_idx=self.param.iters_per_epoch * epoch + batch_idx)
            # for i in range(len(model.optimiser.param_groups)):
            #     model.optimiser.param_groups[i]['lr'] = curr_lr
            #     model.optimiser_classifier_D.param_groups[i]['lr'] = curr_lr

            #### Just for the record, it does not affect the training process
            ssim_value  = pytorch_ssim.ssim(pred_imgs.cuda(), image, window_size=8).detach()
            wandb.log({"Train G: Total loss": loss_G.item(), 
                       "Train G: SCA_Loss": loss_G_img.item(), 
                       "Train G: D": loss_G_tf.item(),  
                       "ssim in train": ssim_value.cpu()})
            wandb.log({"Train CLF: Total loss:": loss_D.item(), 
                       "Train CLF: D_real loss:": loss_D_TF_real.item(), 
                       "Train CLF: D_fake loss:": loss_D_TF_fake.item()})
            # wandb.log({"D loss": errD.item(), "G loss": errG.item()})
            ssim_record = ssim_record + ssim_value.cpu()
            tbar.set_description('idx {}/{} G_Total loss {} G_D loss {} CLF loss {} metrics {}'.format(batch_idx, self.param.iters_per_epoch,
                                                                       loss_G.item(), loss_G_tf.item(), loss_D.item(), ssim_record.numpy()[0]/(batch_idx+1)))


    def train_with_C(self, epoch, data_loader, model):
        tbar = tqdm(range(len(data_loader)), ncols=200)
        data_loader = iter(data_loader)
        
        ssim_record = torch.zeros(1)
        # errD_record = torch.zeros(1)

        for batch_idx in tbar:
            trace, image, prefix, ID = next(data_loader)    
            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            ID = ID.cuda(non_blocking=True)

            #####################################################
            # Train the Discriminator
            model.zero_grad_classifier_C()

            # D_TF real
            embed_real = model.embed(image)

            # D_ID real
            pred_ids_real = model.CLF_C(embed_real)
            loss_D_ID_real = model.ce(pred_ids_real, ID)

            # backward
            loss_D = (loss_D_ID_real).cuda()
            loss_D.backward()
            model.optimiser_classifier_C.step()

            #####################################################
            # Train the Generator
            # train G
            model.zero_grad_encdec()
            
            pred_imgs = model.decoder(model.encoder(trace))
            embed_pred = model.embed(pred_imgs)
            d_id = model.CLF_C(embed_pred)

            loss_G_img = model.loss_fn.loss_function(pred_imgs, image)
            loss_G_img_mse = model.mse(pred_imgs, image)
            loss_G_id = model.ce(d_id, ID)

            # backward
            # loss_G = (100 * (0.84 * loss_G_img + 0.16 * loss_G_img_mse) + loss_G_id).cuda()
            loss_G = (100 * (loss_G_img_mse) + loss_G_id).cuda()
            loss_G.backward()
            model.optimiser.step()

            # # update lr
            # curr_lr = self.poly_lr_decay(curr_idx=self.param.iters_per_epoch * epoch + batch_idx)
            # for i in range(len(model.optimiser.param_groups)):
            #     model.optimiser.param_groups[i]['lr'] = curr_lr
            #     model.optimiser_classifier_C.param_groups[i]['lr'] = curr_lr

            #### Just for the record, it does not affect the training process
            ssim_value  = pytorch_ssim.ssim(pred_imgs.cuda(), image, window_size=8).detach()
            wandb.log({"Train G: Total loss": loss_G.item(), 
                       "Train G: SCA_Loss": loss_G_img.item(), 
                       "Train G: C": loss_G_id.item(), 
                       "ssim in train": ssim_value.cpu()})
            wandb.log({"Train CLF: Total loss:": loss_D.item(), 
                       "Train CLF: C loss:": loss_D_ID_real.item()})
            # wandb.log({"D loss": errD.item(), "G loss": errG.item()})
            ssim_record = ssim_record + ssim_value.cpu()
            tbar.set_description('idx {}/{} G_Total loss {} G_C loss {} CLF loss {} metrics {}'.format(batch_idx, self.param.iters_per_epoch,
                                                                       loss_G.item(), loss_G_id.item(), loss_D.item(), ssim_record.numpy()[0]/(batch_idx+1)))
            
    
    def train_idct_new(self, epoch, data_loader, model):
        
        tbar = tqdm(range(len(data_loader)), ncols=135)
        data_loader = iter(data_loader)

        for batch_idx in tbar:
            trace, image, prefix, ID = next(data_loader)    
            trace = trace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)

            model.zero_grad()
            
            logits = model(trace)

            train_loss = model.bce(logits, image)
            train_loss.backward()
            # model.optim.step()
            model.optimiser.step()
            
            #### Just for the record, it does not affect the training process
            wandb.log({"Train loss": train_loss.item()})

            curr_lr = self.poly_lr_decay(curr_idx=self.param.iters_per_epoch * epoch + batch_idx)
            for i in range(len(model.optimiser.param_groups)):
                model.optimiser.param_groups[i]['lr'] = curr_lr
            tbar.set_description('idx {}/{} loss {} metrics {}'.format(batch_idx, self.param.iters_per_epoch,
                                                                       train_loss.item(), 0))

        return
