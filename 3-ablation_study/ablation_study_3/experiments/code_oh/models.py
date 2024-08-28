from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbam import CBAM


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        m.bias.data.fill_(0)

def weights_init_rnn(model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

class ResDecoder128(nn.Module):
    def __init__(self, dim, nc, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super().__init__()
        self.dim = dim
        self.nc = nc
        self.main = nn.Sequential(
            # state size. (1) x 1 x 1
            nn.ConvTranspose2d(self.dim, self.dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 4 x 4
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 8 x 8
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(self.dim, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        self.apply(weights_init)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.main(input)
        return output

class ImageDecoder(nn.Module):
    def __init__(self, dim, nc, padding_type='reflect', norm_layer=nn.InstanceNorm2d, #nn.BatchNorm2d
                use_dropout=False, use_bias=False):
        super().__init__()
        self.dim = dim
        self.nc = nc
        self.net_part1 = nn.Sequential(
            # state size. (1) x 1 x 1
            nn.ConvTranspose2d(self.dim, self.dim, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 4 x 4
        )
        self.net_part2 = nn.Sequential( # part2
            # state size. (1) x 4 x 4
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 8 x 8
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 16 x 16
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 32 x 32
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim), #nn.BatchNorm2d(self.dim), 
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 64 x 64
            nn.ConvTranspose2d(self.dim, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        self.apply(weights_init)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        tmp = self.net_part1(input)
        output = self.net_part2(tmp)
        return output

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_conv_1d(nn.Module):
    def __init__(self, nin, nout, kernel_size, pool_kernel=4):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size, padding=kernel_size//2),
            nn.MaxPool1d(pool_kernel),
            nn.BatchNorm1d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class attn_trace_encoder_512(nn.Module):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 512 x 512
        self.c1 = dcgan_conv(nc, nf)
        # input is (nf) x 256 x 256
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 128 x 128
        self.c3 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c4 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c5 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c6 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c7 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c8 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.apply(weights_init)

        self.attn1 = CBAM(nf)
        self.attn2 = CBAM(nf)
        self.attn3 = CBAM(nf)
        self.attn4 = CBAM(nf * 2)
        self.attn5 = CBAM(nf * 4)
        self.attn6 = CBAM(nf * 8)
        self.attn7 = CBAM(nf * 8)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.attn5(self.c5(h4))
        h6 = self.attn6(self.c6(h5))
        h7 = self.attn7(self.c7(h6))
        h8 = self.c8(h7)
        return h8.view(-1, self.dim)

    def forward_grad(self, x, grad_saver):
        x = F.normalize(x)
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.attn5(self.c5(h4))
        h6 = self.attn6(self.c6(h5))
        h7 = self.attn7(self.c7(h6))
        h8 = self.c8(h7)
        return h8.view(-1, self.dim)

class trace_encoder_512(nn.Module):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 512 x 512
        self.c1 = dcgan_conv(nc, nf)
        # input is (nf) x 256 x 256
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 128 x 128
        self.c3 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c4 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c5 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c6 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c7 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c8 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.apply(weights_init)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        h7 = self.c7(h6)
        h8 = self.c8(h7)
        return h8.view(-1, self.dim)

class attn_trace_encoder_256(nn.Module):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 256 x 256
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 128 x 128
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c3 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c4 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c5 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c6 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c7 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.apply(weights_init)

        self.attn1 = CBAM(nf)
        self.attn2 = CBAM(nf)
        self.attn3 = CBAM(nf * 2)
        self.attn4 = CBAM(nf * 4)
        self.attn5 = CBAM(nf * 8)
        self.attn6 = CBAM(nf * 8)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.attn5(self.c5(h4))
        h6 = self.attn6(self.c6(h5))
        h7 = self.c7(h6)
        return h7.view(-1, self.dim)

    def forward_grad(self, x, grad_saver):
        x = F.normalize(x)
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.attn5(self.c5(h4))
        h6 = self.attn6(self.c6(h5))
        h7 = self.c7(h6)
        return h7.view(-1, self.dim)

class TraceEncoder_1DCNN(nn.Module):
    """
    Modified from attn_trace_encoder_256
    Accept input: (batch size x 393,216)
    """
    def __init__(self, input_len, dim):
        super().__init__()
        self.dim = dim
        nf = 64
        self.c0 = nn.Flatten()
        self.network = nn.Sequential(
            # batch size x 1 x (6 x 256 x 256 = 393,216)
            dcgan_conv_1d(1, nf, 96),
            # batch size x nf x 98304
            dcgan_conv_1d(nf, nf, 4),
            # batch size x nf x 24576
            dcgan_conv_1d(nf, nf * 2, 4),
            # batch size x nf*2 x 6144
            dcgan_conv_1d(nf * 2, nf * 4, 4),
            # batch size x nf*4 x 1536
            dcgan_conv_1d(nf * 4, nf * 8, 4),
            # batch size x nf*8 x 384
            dcgan_conv_1d(nf * 8, nf * 8, 4),
            # batch size x nf*8 x 96
            # self.c7 = nn.Sequential(
            #         nn.Conv1d(nf * 8, dim, 96), 
            #         # batch size x dim x 1
            #         nn.BatchNorm1d(dim),
            #         nn.Tanh()
            #         )
            nn.Sequential(
                nn.Flatten(),
                # batch size x nf*8*96
                nn.Linear(nf*8*96, 128),
                # batch size x 128
                nn.BatchNorm1d(128),
            )
        )

        self.apply(weights_init)

    def forward(self, x):
        x = self.c0(F.normalize(x))
        x = torch.unsqueeze(x, dim=1)
        h7 = self.network(x)
        return h7.view(-1, self.dim)

class TraceEncoder_1DCNN_encode(nn.Module):
    """
    Accept input: (batch size x input_len x 64)
    """
    def __init__(self, input_len, dim):
        super().__init__()
        self.dim = dim
        nf = 64
        l = floor(floor(floor(floor(floor(floor(input_len / 4) / 4) / 4) / 4) / 4) / 4)
        self.network = nn.Sequential(
            # batch size x 64 x input_len
            dcgan_conv_1d(64, nf, 6),
            # batch size x nf x (input_len / 4)
            dcgan_conv_1d(nf, nf, 4),
            # batch size x nf x (input_len / 4 / 4)
            dcgan_conv_1d(nf, nf * 2, 4),
            # batch size x nf*2 x (input_len / 4 / 4 / 4)
            dcgan_conv_1d(nf * 2, nf * 4, 4),
            # batch size x nf*4 x (input_len / 4 / 4 / 4 / 4)
            dcgan_conv_1d(nf * 4, nf * 8, 4),
            # batch size x nf*8 x (input_len / 4 / 4 / 4 / 4 / 4)
            dcgan_conv_1d(nf * 8, nf * 8, 4),
            # batch size x nf*8 x (input_len / 4 / 4 / 4 / 4 / 4 / 4)
            nn.Sequential(
                nn.Flatten(),
                # batch size x nf*8*l
                nn.Linear(nf*8*l, 128),
                # batch size x 128
                nn.BatchNorm1d(128),
            )
        )

        self.apply(weights_init)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        return out.view(-1, self.dim)


class TraceEncoder_1DCNN_encode_attn(nn.Module):
    """
    Accept input: (batch size x input_len x 64)
    """
    def __init__(self, input_len, dim):
        super().__init__()
        self.dim = dim
        nf = 64
        l = floor(floor(floor(floor(floor(floor(input_len / 4) / 4) / 4) / 4) / 4) / 4)

        self.c1 = dcgan_conv_1d(64, nf, 6)
        self.c2 = dcgan_conv_1d(nf, nf, 4)
        self.c3 = dcgan_conv_1d(nf, nf * 2, 4)
        self.c4 = dcgan_conv_1d(nf * 2, nf * 4, 4)
        self.c5 = dcgan_conv_1d(nf * 4, nf * 8, 4)
        self.c6 = dcgan_conv_1d(nf * 8, nf * 8, 4)
        self.c7 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nf*8*l, 128),
                nn.BatchNorm1d(128),
            )

        self.apply(weights_init)

        self.attn1 = CBAM(nf)
        self.attn2 = CBAM(nf)
        self.attn3 = CBAM(nf * 2)
        self.attn4 = CBAM(nf * 4)
        self.attn5 = CBAM(nf * 8)
        self.attn6 = CBAM(nf * 8)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        c1x = self.c1(x).unsqueeze(2)
        h1 = self.attn1(c1x).squeeze(2)
        
        
        exit(1)

        # h2 = self.attn2(self.c2(h1))






    #     self.network = nn.Sequential(
    #         # batch size x 64 x input_len
    #         dcgan_conv_1d(64, nf, 6),



    #         # batch size x nf x (input_len / 4)
    #         dcgan_conv_1d(nf, nf, 4),
    #         # batch size x nf x (input_len / 4 / 4)
    #         dcgan_conv_1d(nf, nf * 2, 4),
    #         # batch size x nf*2 x (input_len / 4 / 4 / 4)
    #         dcgan_conv_1d(nf * 2, nf * 4, 4),
    #         # batch size x nf*4 x (input_len / 4 / 4 / 4 / 4)
    #         dcgan_conv_1d(nf * 4, nf * 8, 4),
    #         # batch size x nf*8 x (input_len / 4 / 4 / 4 / 4 / 4)
    #         dcgan_conv_1d(nf * 8, nf * 8, 4),
    #         # batch size x nf*8 x (input_len / 4 / 4 / 4 / 4 / 4 / 4)
    #         nn.Sequential(
    #             nn.Flatten(),
    #             # batch size x nf*8*l
    #             nn.Linear(nf*8*l, 128),
    #             # batch size x 128
    #             nn.BatchNorm1d(128),
    #         )
    #     )

    #     self.apply(weights_init)

    # def forward(self, x):
    #     x = x.permute(0, 2, 1)
    #     out = self.network(x)
    #     return out.view(-1, self.dim)

class TraceEncoder_1DCNN_encode_webp(nn.Module):
    """
    Trace Encoder for WebP
    Accept input: (batch size x input_len x 64)
    """
    def __init__(self, input_len, dim):
        super().__init__()
        self.dim = dim
        nf = 64
        l = floor(floor(floor(floor(floor(floor(input_len / 6) / 4) / 4) / 4) / 4) / 4)
        self.network = nn.Sequential(
            # batch size x 64 x input_len
            dcgan_conv_1d(64, nf, 6, 6),
            # batch size x nf x (input_len / 6)
            dcgan_conv_1d(nf, nf, 4),
            # batch size x nf x (input_len / 6 / 4)
            dcgan_conv_1d(nf, nf * 2, 4),
            # batch size x nf*2 x (input_len / 6 / 4 / 4)
            dcgan_conv_1d(nf * 2, nf * 4, 4),
            # batch size x nf*4 x (input_len / 6 / 4 / 4 / 4)
            dcgan_conv_1d(nf * 4, nf * 8, 4),
            # batch size x nf*8 x (input_len / 6 / 4 / 4 / 4 / 4)
            dcgan_conv_1d(nf * 8, nf * 8, 4),
            # batch size x nf*8 x (input_len / 6 / 4 / 4 / 4 / 4 / 4)
            nn.Sequential(
                nn.Flatten(),
                # batch size x nf*8*l
                nn.Linear(nf*8*l, 128),
                # batch size x 128
                nn.BatchNorm1d(128),
            )
        )

        self.apply(weights_init)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        return out.view(-1, self.dim)

class trace_encoder_256(nn.Module):
    """
    Accept input: (batch size) x 6 x 256 x 256
    or length 393,216
    """
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 256 x 256
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 128 x 128
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c3 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c4 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c5 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c6 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c7 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        h7 = self.c7(h6)

        return h7.view(-1, self.dim)

class TraceEncoder(nn.Module):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 256 x 256
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 128 x 128
        self.c2 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c3 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c4 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c5 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c6 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c7 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nf * 128, nf * 128)
        )
        self.fc_mu = nn.Linear(nf * 128, dim)
        self.fu_logvar = nn.Linear(nf * 128, dim)

        self.apply(weights_init)

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        h7 = self.c7(h6)

        mu = self.fc_mu(h7)
        log_var = self.fu_logvar(h7)
        output = self.reparameterize(mu, log_var)
        
        return mu, log_var, output

class attn_trace_encoder_square_128(nn.Module):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        # self.c5 = dcgan_conv(nf * 8, nf * 8)
        # # state size. (nf*8) x 4 x 4
        # self.c6 = nn.Sequential(
        #         nn.Conv2d(nf * 8, dim, 4, 1, 0),
        #         nn.BatchNorm2d(dim),
        #         nn.Tanh()
        #         )
        self.apply(weights_init)
        self.attn1 = CBAM(nf)
        self.attn2 = CBAM(nf * 2)
        self.attn3 = CBAM(nf * 4)
        self.attn4 = CBAM(nf * 8)

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.c5(h4)
        #h6 = self.c6(h5)
        return h5.transpose(1, 2).transpose(2, 3)

    def forward_grad(self, x, grad_saver):
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        x = F.normalize(x)
        h1 = self.attn1(self.c1(x))
        h2 = self.attn2(self.c2(h1))
        h3 = self.attn3(self.c3(h2))
        h4 = self.attn4(self.c4(h3))
        h5 = self.c5(h4)
        #h6 = self.c6(h5)
        return h5.transpose(1, 2).transpose(2, 3)

class RefinerD(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # state size. (nc) x 128 x 128
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)

class RefinerG_BN(nn.Module):
    def __init__(self, nc, ngf):
        super().__init__()

        self.nc = nc
        self.ngf = ngf
        # 128
        self.net1 = nn.Sequential(
                    nn.Conv2d(nc, ngf, 4, 2, 1)
                    )
        self.net2 = nn.Sequential(
                    nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 2)
                    )
        self.net3 = nn.Sequential(
                    nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 4)
                    )
        self.net4 = nn.Sequential(
                    nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.net5 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.net6 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.net7 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
                    )

        self.dnet1 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet2 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet3 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet4 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 4, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 4)
                    )
        self.dnet5 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 4 , ngf * 2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 2)
                    )
        self.dnet6 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2 , ngf, 4, 2, 1),
                    nn.BatchNorm2d(ngf)
                    )
        self.dnet7 = nn.Sequential(
                    nn.ConvTranspose2d(ngf , nc, 4, 2, 1)
                    )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 128 x 128
        e1 = self.net1(input)
        # state size is (ngf) x 64 x 64
        e2 = self.net2(self.leaky_relu(e1))
        # state size is (ngf x 2) x 32 x 32
        e3 = self.net3(self.leaky_relu(e2))
        # state size is (ngf x 4) x 16 x 16
        e4 = self.net4(self.leaky_relu(e3))
        # state size is (ngf x 8) x 8 x 8
        e5 = self.net5(self.leaky_relu(e4))
        # state size is (ngf x 8) x 4 x 4
        e6 = self.net6(self.leaky_relu(e5))
        # state size is (ngf x 8) x 2 x 2
        e7 = self.net7(self.leaky_relu(e6))
        # state size is (ngf x 8) x 1 x 1
        # No batch norm on output of Encoder

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1 = self.dropout(self.dnet1(self.relu(e7)))
        # state size is (ngf x 8) x 2 x 2
        d2 = self.dropout(self.dnet2(self.relu(d1)))
        # state size is (ngf x 8) x 4 x 4
        d3 = self.dropout(self.dnet3(self.relu(d2)))
        # state size is (ngf x 8) x 8 x 8
        d4 = self.dnet4(self.relu(d3))
        # state size is (ngf x 4) x 16 x 16
        d5 = self.dnet5(self.relu(d4))
        # state size is (ngf x 2) x 32 x 32
        d6 = self.dnet6(self.relu(d5))
        # state size is (ngf) x 64 x 64
        d7 = self.dnet7(self.relu(d6))
        # state size is (nc) x 128 x 128
        output = self.tanh(d7)
        return output

class ImageEncoder(nn.Module):
    def __init__(self, nc, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super().__init__()
        self.nc = nc
        self.dim = dim
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(self.nc, self.dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 64 x 64
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 32 x 32
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 16 x 16
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 8 x 8
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (dim) x 4 x 4
            # nn.Conv2d(self.dim, self.dim, 4, 1, 0, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim) x 1 x 1
        )
        self.fc_mu = nn.Linear(dim*16, dim)
        self.fc_var = nn.Linear(dim*16, dim)
        self.apply(weights_init)

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        output = self.main(input)
        output = torch.flatten(output, start_dim=1)
        mu = self.fc_mu(output)
        log_var = self.fc_var(output)
        output = self.reparameterize(mu, log_var)

        return mu, log_var, output

class image_decoder_128(nn.Module):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.ReLU()
                )
        # state size. (nf*8) x 4 x 2
        self.upc2 = dcgan_upconv(nf * 8, nf * 8)
        # state size. (nf*8) x 8 x 4
        self.upc3 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 16 x 8
        self.upc4 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 32 x 16
        self.upc5 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 64 x 32
        self.upc6 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Tanh() # --> [-1, 1]
                #nn.Sigmoid()
                # state size. (nc) x 128 x 64
                )
        self.apply(weights_init)

    def forward(self, x):
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        d5 = self.upc5(d4)
        d6 = self.upc6(d5)
        return d6

class image_output_embed_128(nn.Module):
    def __init__(self, dim=128, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 128 x 128
        self.c1 = nn.Sequential(
                nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                #nn.Sigmoid()
                )
        self.apply(weights_init)

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

class discriminator_128(nn.Module):
    def __init__(self, dim=1, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 128 x 128
        self.c1 = nn.Sequential(
                nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.Sigmoid()
                )
        self.apply(weights_init)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            # [bs, 128, 128] --> [bs, 1, 128, 128]
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

class classifier(nn.Module):
    def __init__(self, dim, n_class, use_bn=False):
        super().__init__()
        self.dim = dim
        self.n_class = n_class
        if use_bn:
            self.main = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),

                    nn.Linear(dim, n_class),
                    nn.Sigmoid()
                )
        else:
            self.main = nn.Sequential(
                    nn.Linear(dim, n_class),
                    nn.Sigmoid()
                )
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, self.dim)
        out = self.main(x)
        return out

class Generator(nn.Module):
    def __init__(self, nc, dim):
        super().__init__()

        self.nc = nc
        self.dim = dim
        # 128
        self.net1 = nn.Sequential(
                    nn.Conv2d(nc, dim, 4, 2, 1)
                    )
        self.net2 = nn.Sequential(
                    nn.Conv2d(dim, dim * 2, 4, 2, 1),
                    nn.BatchNorm2d(dim * 2)
                    )
        self.net3 = nn.Sequential(
                    nn.Conv2d(dim * 2, dim * 4, 4, 2, 1),
                    nn.BatchNorm2d(dim * 4)
                    )
        self.net4 = nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.net5 = nn.Sequential(
                    nn.Conv2d(dim * 8, dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.net6 = nn.Sequential(
                    nn.Conv2d(dim * 8, dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.net7 = nn.Sequential(
                    nn.Conv2d(dim * 8, dim * 8, 4, 2, 1),
                    #nn.BatchNorm2d(dim * 8)
                    )

        self.dnet1 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 8 , dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.dnet2 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 8 , dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.dnet3 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 8 , dim * 8, 4, 2, 1),
                    nn.BatchNorm2d(dim * 8)
                    )
        self.dnet4 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 8 , dim * 4, 4, 2, 1),
                    nn.BatchNorm2d(dim * 4)
                    )
        self.dnet5 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 4 , dim * 2, 4, 2, 1),
                    nn.BatchNorm2d(dim * 2)
                    )
        self.dnet6 = nn.Sequential(
                    nn.ConvTranspose2d(dim * 2 , dim, 4, 2, 1),
                    nn.BatchNorm2d(dim)
                    )
        self.dnet7 = nn.Sequential(
                    nn.ConvTranspose2d(dim , nc, 4, 2, 1)
                    )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # input is (nc) x 128 x 128
        e1 = self.net1(input)
        # state size is (dim) x 64 x 64
        e2 = self.net2(self.leaky_relu(e1))
        # state size is (dim x 2) x 32 x 32
        e3 = self.net3(self.leaky_relu(e2))
        # state size is (dim x 4) x 16 x 16
        e4 = self.net4(self.leaky_relu(e3))
        # state size is (dim x 8) x 8 x 8
        e5 = self.net5(self.leaky_relu(e4))
        # state size is (dim x 8) x 4 x 4
        e6 = self.net6(self.leaky_relu(e5))
        # state size is (dim x 8) x 2 x 2
        e7 = self.net7(self.leaky_relu(e6))
        # state size is (dim x 8) x 1 x 1
        d1 = self.dropout(self.dnet1(self.relu(e7)))
        # state size is (dim x 8) x 2 x 2
        d2 = self.dropout(self.dnet2(self.relu(d1)))
        # state size is (dim x 8) x 4 x 4
        d3 = self.dropout(self.dnet3(self.relu(d2)))
        # state size is (dim x 8) x 8 x 8
        d4 = self.dnet4(self.relu(d3))
        # state size is (dim x 4) x 16 x 16
        d5 = self.dnet5(self.relu(d4))
        # state size is (dim x 2) x 32 x 32
        d6 = self.dnet6(self.relu(d5))
        # state size is (dim) x 64 x 64
        d7 = self.dnet7(self.relu(d6))
        # state size is (nc) x 128 x 128
        output = self.tanh(d7)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc, dim):
        super().__init__()
        self.nc = nc
        self.dim = dim
        self.main = nn.Sequential(
            # state size. (nc) x 128 x 128
            nn.Conv2d(self.nc, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim) x 64 x 64
            nn.Conv2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim) x 32 x 32
            nn.Conv2d(self.dim, self.dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim*2) x 16 x 16
            nn.Conv2d(self.dim * 2, self.dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim*4) x 8 x 8
            nn.Conv2d(self.dim * 4, self.dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dim*8) x 4 x 4
            nn.Conv2d(self.dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)

# class TraceEncoderLSTM(nn.Module):
#     def __init__(self, output_dim):
#         super().__init__()
#         self.network1 = nn.Sequential(
#             # batch_size x cache_lines(64) x length(8192)
#             nn.Conv1d(64, 64, 3),
#             nn.ReLU(),
#             # batch_size x features(64) x length(8190)
#             nn.MaxPool1d(4),
#             # batch_size x features(64) x length(2048)
#             nn.Conv1d(64, 64, 3),
#             nn.ReLU(),
#             # batch_size x features(64) x length(2048)
#             nn.MaxPool1d(4),
#             # batch_size x features(64) x length(511)
#         )
#         # batch_size x length(511) x features(64)
#         self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
#         self.network2 = nn.Sequential(
#             # batch_size x length(511) x hidden_states(32)
#             nn.Flatten(),
#             nn.Dropout(0.2),
#             nn.Linear(511*32, output_dim)
#             # batch_size x output_dim(128)
#         )
#         self.apply(weights_init)

#     def forward(self, trace):
#         c1 = trace.permute(0, 2, 1)
#         c2 = self.network1(c1)
#         c2 = c2.permute(0, 2, 1)
#         c3, _ = self.lstm(c2)
#         output = self.network2(c3)
#         return output

class TraceEncoderLSTM(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.network1 = nn.Sequential(
            # batch_size x cache_lines(64) x length(5600)
            nn.Conv1d(64, 64, 3),
            nn.ReLU(),
            # batch_size x features(64) x length(5598)
            nn.MaxPool1d(4),
            # batch_size x features(64) x length(1399)
            nn.Conv1d(64, 64, 3),
            nn.ReLU(),
            # batch_size x features(64) x length(1397)
            nn.MaxPool1d(4),
            # batch_size x features(64) x length(349)
        )
        # batch_size x length(349) x features(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.network2 = nn.Sequential(
            # batch_size x length(349) x hidden_states(32)
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(349*32, output_dim)
            # batch_size x output_dim(128)
        )
        self.apply(weights_init)

    def forward(self, trace):
        c1 = trace.permute(0, 2, 1)
        c2 = self.network1(c1)
        c2 = c2.permute(0, 2, 1)
        c3, _ = self.lstm(c2)
        output = self.network2(c3)
        return output

class TraceEncoderLSTMPin(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.network1 = nn.Sequential(
            # batch_size x 1 x length(300,000)
            nn.Conv1d(1, 256, 16, stride=3),
            nn.ReLU(),
            # batch_size x features(256) x length(99,995)
            nn.MaxPool1d(4),
            # batch_size x features(256) x length(24998)
            nn.Conv1d(256, 256, 8, stride=3),
            nn.ReLU(),
            # batch_size x features(256) x length(8331)
            nn.MaxPool1d(4),
            # batch_size x features(256) x length(2082)
        )
        # batch_size x length(2082) x features(256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=32, batch_first=True)
        self.network2 = nn.Sequential(
            # batch_size x length(2082) x hidden_states(32)
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(2082*32, output_dim)
            # batch_size x output_dim(128)
        )
        self.apply(weights_init)

    def forward(self, trace):
        c1 = torch.unsqueeze(trace, dim=1)
        c2 = self.network1(c1)
        c2 = c2.permute(0, 2, 1)
        c3, _ = self.lstm(c2)
        output = self.network2(c3)
        return output
