import torch.nn as nn
import torch
# from model.encoder.encoder import TraceEncoder_1DCNN_encode_384_64, TraceEncoder_1DCNN_encode_24576_64, TraceEncoder_1DCNN_encode_300000_64 #TraceEncoder_1DCNN_encode
import model.encoder.encoder as enc
from model.decoder.decoder import ResDecoder128
import model.classifier.classifier as gan_clf
import loss.loss as loss

import model.encoder.pp_encoder as pp_enc
import model.decoder.pp_decoder as pp_dec

class MyModel(nn.Module):
    def __init__(self, configs):
        super(MyModel, self).__init__()
        # self.encoder = TraceEncoder_1DCNN_encode(input_len=configs.trace_lens, dim=configs.latent_dims)
        self.encoder = enc.TraceEncoder_1DCNN_encode_384_64(input_len=configs.trace_lens, dim=configs.latent_dims)
        if configs.trace_lens == 384:
            if configs.dim == 2:
                if configs.attn:
                    self.encoder = enc.ATTN_TraceEncoder_2DCNN_encode_384_64(input_len=configs.trace_lens, dim=configs.latent_dims)
                else:
                    self.encoder = enc.TraceEncoder_2DCNN_encode_384_64(input_len=configs.trace_lens, dim=configs.latent_dims)
            else:
                self.encoder = enc.TraceEncoder_1DCNN_encode_384_64(input_len=configs.trace_lens, dim=configs.latent_dims)

        elif configs.trace_lens == 24576:
            if configs.dim == 2:
                if configs.attn:
                    self.encoder = enc.ATTN_TraceEncoder_2DCNN_encode_24576_64(input_len=configs.trace_lens, dim=configs.latent_dims)
                else:
                    self.encoder = enc.TraceEncoder_2DCNN_encode_24576_64(input_len=configs.trace_lens, dim=configs.latent_dims)
            else:
                self.encoder = enc.TraceEncoder_1DCNN_encode_24576_64(input_len=configs.trace_lens, dim=configs.latent_dims)
        
        elif configs.trace_lens == 300000:
            self.encoder = enc.TraceEncoder_1DCNN_encode_300000_64(input_len=configs.trace_lens, dim=configs.latent_dims)
        
        elif configs.trace_lens == 800:
            if configs.dim == 2:
                if configs.attn:
                    self.encoder = pp_enc.ATTN_PP_TraceEncoder_2DCNN_encode_800_64(input_len=configs.trace_lens, dim=configs.latent_dims)
                else:
                    self.encoder = pp_enc.PP_TraceEncoder_2DCNN_encode_800_64(input_len=configs.trace_lens, dim=configs.latent_dims)
            else:
                self.encoder = pp_enc.PP_TraceEncoder_1DCNN_encode_800_64(input_len=configs.trace_lens, dim=configs.latent_dims)

        if configs.target == "img":
            self.decoder = ResDecoder128(dim=configs.latent_dims, nc=3)
        elif configs.target == "idct":
            print("Decoder: IDCT")
            self.decoder = pp_dec.PP_Trace2IDCT_01_DecNET(dim=configs.latent_dims, nc=384)

        ## GAN Components
        # self.classifier = img_classifier(dim=128, n_class=1, use_bn=False)

        self.embed = gan_clf.image_output_embed_128(dim=128, nc=3)
        self.CLF_D = gan_clf.CLF_D(dim=128)
        self.CLF_C = gan_clf.CLF_C(dim=128, CLF_NUM=10177)

        self.loss_fn = loss.SCA_Loss()

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

        # self.optimiser = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=configs.lr, betas=configs.betas, weight_decay=configs.weight_decay)
        #self.optimiser_classifier = torch.optim.Adam(list(self.classifier.parameters()) + list(self.embed.parameters()), lr=configs.lr, betas=configs.betas, weight_decay=configs.weight_decay)
        # self.optimiser_classifier = torch.optim.Adam(list(self.embed.parameters()) + \
        #                                 list(self.CLF_D.parameters()) + \
        #                                 list(self.CLF_C.parameters()),
        #                                 lr=configs.lr, betas=configs.betas, weight_decay=configs.weight_decay)
        # self.optimiser_classifier_D = torch.optim.Adam(list(self.embed.parameters()) + \
        #                                 list(self.CLF_D.parameters()),
        #                                 lr=configs.lr, betas=configs.betas, weight_decay=configs.weight_decay)
        # self.optimiser_classifier_C = torch.optim.Adam(list(self.embed.parameters()) + \
        #                                 list(self.CLF_C.parameters()),
        #                                 lr=configs.lr, betas=configs.betas, weight_decay=configs.weight_decay)
        
        #self.optim = torch.optim.Adam(list(self.parameters()), lr=2e-4, betas=(0.5, 0.999))
        #self.optim = torch.optim.Adam(list(self.parameters()), lr=configs.lr, betas=configs.betas, weight_decay=configs.weight_decay)

        self.optimiser = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=configs.lr, betas=configs.betas)
        self.optimiser_classifier = torch.optim.Adam(list(self.embed.parameters()) + \
                                        list(self.CLF_D.parameters()) + \
                                        list(self.CLF_C.parameters()),
                                        lr=configs.lr, betas=configs.betas)
        self.optimiser_classifier_D = torch.optim.Adam(list(self.embed.parameters()) + \
                                        list(self.CLF_D.parameters()),
                                        lr=configs.lr, betas=configs.betas)
        self.optimiser_classifier_C = torch.optim.Adam(list(self.embed.parameters()) + \
                                        list(self.CLF_C.parameters()),
                                        lr=configs.lr, betas=configs.betas)



    def zero_grad_encdec(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def zero_grad_classifier(self):
        # self.classifier.zero_grad()
        # self.embed.zero_grad()
        self.embed.zero_grad()
        self.CLF_D.zero_grad()
        self.CLF_C.zero_grad()

    def zero_grad_classifier_D(self):
        # self.classifier.zero_grad()
        # self.embed.zero_grad()
        self.embed.zero_grad()
        self.CLF_D.zero_grad()

    def zero_grad_classifier_C(self):
        # self.classifier.zero_grad()
        # self.embed.zero_grad()
        self.embed.zero_grad()
        self.CLF_C.zero_grad()

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
    
    
