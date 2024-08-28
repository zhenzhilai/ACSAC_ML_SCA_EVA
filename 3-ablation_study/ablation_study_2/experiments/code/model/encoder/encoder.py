from math import floor

import torch
import torch.nn as nn
from model.encoder.cbam import CBAM

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





    
###########################################################
##### This is for 24576 * 64, simple CNN
class TraceEncoder_1DCNN_encode_24576_64(nn.Module):
    """
    Accept input: (batch size x input_len x 64)
    """
    def __init__(self, input_len, dim):
        super().__init__()
        self.dim = dim
        nf = 64

        # # stride
        # self.network = nn.Sequential(
        #     nn.Conv1d(64, nf, kernel_size=7, stride=2, padding=3),  
        #     # nn.BatchNorm1d(nf),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv1d(nf, nf * 2, kernel_size=5, stride=2, padding=2),
        #     # nn.BatchNorm1d(nf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv1d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm1d(nf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Flatten(),
        #     nn.Linear(nf * 4 * (384 // 8), 128),
        #     nn.BatchNorm1d(128),
        # )

        ## stride bm
        # self.network = nn.Sequential(
        #     dcgan_conv_1d_stridebm(64, nf, 7, 3),
        #     dcgan_conv_1d_stridebm(nf, nf * 2, 5, 2),
        #     dcgan_conv_1d_stridebm(nf * 2, nf * 4, 3, 1),

        #     nn.Flatten(),
        #     nn.Linear(nf * 4 * (24576 // 8), 128),
        #     nn.BatchNorm1d(128),
        # )

        ## maxpool k=4
        self.network = nn.Sequential(
            dcgan_conv_1d(64, nf, 6),
            # batch size x nf x (input_len / 4)
            dcgan_conv_1d(nf, nf, 4),
            # batch size x nf x (input_len / 4 / 4)
            dcgan_conv_1d(nf, nf * 2, 4),
            # batch size x nf*2 x (input_len / 4 / 4 / 4)
            # dcgan_conv_1d(nf * 2, nf * 4, 4),

            nn.Flatten(),
            nn.Linear(nf * 2 * (24576 // (4*4*4)), 128),
            nn.BatchNorm1d(128),
        )

        # ## maxpool
        # l = floor(floor(floor(input_len / 2) / 2) / 2)  

        # self.network = nn.Sequential(
        #     # batch size x 64 x input_len
        #     dcgan_conv_1d_24576_64(64, nf, 6, pool_kernel=2),
        #     # batch size x nf x (input_len / 2)
        #     dcgan_conv_1d_24576_64(nf, nf * 2, 4, pool_kernel=2),
        #     # batch size x nf*2 x (input_len / 2 / 2)
        #     dcgan_conv_1d_24576_64(nf * 2, nf * 4, 4, pool_kernel=2),
        #     # batch size x nf*4 x (input_len / 2 / 2 / 2)
        #     nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(nf*4*l, 128),  
        #         nn.BatchNorm1d(128),
        #     )
        # )

        self.apply(weights_init)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # print(out.shape)
        # exit(1)
        return out.view(-1, self.dim)


class dcgan_conv_1d_24576_64(nn.Module):
    def __init__(self, nin, nout, kernel_size, pool_kernel=4):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size, padding=kernel_size // 2),
            nn.MaxPool1d(pool_kernel),
            nn.BatchNorm1d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)
    
class dcgan_conv_1d_stridebm(nn.Module):
    def __init__(self, nin, nout, kernel_size, pad):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=2, padding=pad),  
            nn.BatchNorm1d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


#####################################################################
#### This is for 384 * 64
class TraceEncoder_1DCNN_encode_384_64(nn.Module):
    """
    Accept input: (batch size x input_len x 64)
    """
    def __init__(self, input_len, dim):
        super().__init__()
        self.dim = dim
        nf = 64


        ## stride
        # self.network = nn.Sequential(
        #     nn.Conv1d(64, nf, kernel_size=7, stride=2, padding=3),  
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv1d(nf, nf * 2, kernel_size=5, stride=2, padding=2),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv1d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Flatten(),
        #     nn.Linear(nf * 4 * (384 // 8), 128),
        #     nn.BatchNorm1d(128),
        # )

        ## stride bm
        # self.network = nn.Sequential(
        #     dcgan_conv_1d_stridebm(64, nf, 7, 3),
        #     dcgan_conv_1d_stridebm(nf, nf * 2, 5, 2),
        #     dcgan_conv_1d_stridebm(nf * 2, nf * 4, 3, 1),

        #     nn.Flatten(),
        #     nn.Linear(nf * 4 * (384 // 8), 128),
        #     nn.BatchNorm1d(128),
        # )

        ## maxpool k=4
        self.network = nn.Sequential(
            dcgan_conv_1d(64, nf, 6),
            # batch size x nf x (input_len / 4)
            dcgan_conv_1d(nf, nf, 4),
            # batch size x nf x (input_len / 4 / 4)
            dcgan_conv_1d(nf, nf * 2, 4),

            nn.Flatten(),
            nn.Linear(nf * 2 * (384 // (4*4*4)), 128),
            nn.BatchNorm1d(128),
        )

        ## maxpool
        # l = floor(floor(floor(input_len / 2) / 2) / 2)  

        # self.network = nn.Sequential(
        #     # batch size x 64 x input_len
        #     dcgan_conv_1d_24576_64(64, nf, 6, pool_kernel=2),
        #     # batch size x nf x (input_len / 2)
        #     dcgan_conv_1d_24576_64(nf, nf * 2, 4, pool_kernel=2),
        #     # batch size x nf*2 x (input_len / 2 / 2)
        #     dcgan_conv_1d_24576_64(nf * 2, nf * 4, 4, pool_kernel=2),
        #     # batch size x nf*4 x (input_len / 2 / 2 / 2)
        #     nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(nf*4*l, 128),  
        #         nn.BatchNorm1d(128),
        #     )
        # )

        self.apply(weights_init)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # print(out.shape)
        # exit(1)
        return out.view(-1, self.dim)
    



#####################################################################
#### Below is the old version of TraceEncoder_1DCNN_encode, from the paper
class dcgan_conv_1d(nn.Module):
    def __init__(self, nin, nout, kernel_size, pool_kernel=4):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size, padding=kernel_size // 2),
            nn.MaxPool1d(pool_kernel),
            nn.BatchNorm1d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)

class TraceEncoder_1DCNN_encode_300000_64(nn.Module):
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



#####################################################################
#####################################################################
#####################################################################
##################### 2D Encoder 
#####################################################################
#####################################################################


class TraceEncoder_2DCNN_encode_384_64(torch.nn.Module):
    def __init__(self, input_len, dim):
        super(TraceEncoder_2DCNN_encode_384_64, self).__init__()

        self.dim = dim
        self.nf = 16
        self.nc = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.nc, 
                      out_channels = self.nf, 
                      kernel_size = (4,4), 
                      stride = (2, 2), 
                      padding = 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf, 
                      out_channels = self.nf, 
                      kernel_size = (4,4), 
                      stride = (2, 2), 
                      padding = 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf, 
                      out_channels = self.nf*2, 
                      kernel_size = (4,4), 
                      stride = 2, 
                      padding = 1),
            nn.BatchNorm2d(self.nf*2),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*2, 
                      out_channels = self.nf*4, 
                      kernel_size = (4,4), 
                      stride = 2, 
                      padding = 1),
            nn.BatchNorm2d(self.nf*4),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*4, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))

        
        self.convFinal = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.dim, 
                      kernel_size = (6,4), 
                      stride = (2, 1), 
                      padding = 0),
            nn.Tanh())

        self.apply(weights_init)    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.convFinal(x)
        x = x.view(-1, self.nf*8)

        return x
    


class TraceEncoder_2DCNN_encode_24576_64(torch.nn.Module):
    def __init__(self, input_len, dim):
        super(TraceEncoder_2DCNN_encode_24576_64, self).__init__()

        self.dim = dim
        self.nf = 16
        self.nc = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.nc, 
                      out_channels = self.nf, 
                      kernel_size = (4,4), 
                      stride = (2, 2), 
                      padding = 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf, 
                      out_channels = self.nf, 
                      kernel_size = (4,4), 
                      stride = (2, 2), 
                      padding = 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf, 
                      out_channels = self.nf*2, 
                      kernel_size = (4,4), 
                      stride = 2, 
                      padding = 1),
            nn.BatchNorm2d(self.nf*2),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*2, 
                      out_channels = self.nf*4, 
                      kernel_size = (4,4), 
                      stride = 2, 
                      padding = 1),
            nn.BatchNorm2d(self.nf*4),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*4, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))

        
        self.convFinal = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.dim, 
                      kernel_size = (6,4), 
                      stride = (1, 1), 
                      padding = 0),
            nn.Tanh())

        self.apply(weights_init)    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.convFinal(x)
        x = x.view(-1, self.nf*8)

        return x
    


class ATTN_TraceEncoder_2DCNN_encode_384_64(torch.nn.Module):
    def __init__(self, input_len, dim):
        super(ATTN_TraceEncoder_2DCNN_encode_384_64, self).__init__()

        self.dim = dim
        self.nf = 16
        self.nc = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.nc, 
                      out_channels = self.nf, 
                      kernel_size = (4,4), 
                      stride = (2, 2), 
                      padding = 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf, 
                      out_channels = self.nf, 
                      kernel_size = (4,4), 
                      stride = (2, 2), 
                      padding = 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf, 
                      out_channels = self.nf*2, 
                      kernel_size = (4,4), 
                      stride = 2, 
                      padding = 1),
            nn.BatchNorm2d(self.nf*2),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*2, 
                      out_channels = self.nf*4, 
                      kernel_size = (4,4), 
                      stride = 2, 
                      padding = 1),
            nn.BatchNorm2d(self.nf*4),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*4, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))

        
        self.convFinal = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.dim, 
                      kernel_size = (6,4), 
                      stride = (2, 1), 
                      padding = 0),
            nn.Tanh())
        self.apply(weights_init)    
        
        self.attn1 = CBAM(self.nf)
        self.attn2 = CBAM(self.nf)
        self.attn3 = CBAM(self.nf * 2)
        self.attn4 = CBAM(self.nf * 4)
        self.attn5 = CBAM(self.nf * 8)
        self.attn6 = CBAM(self.nf * 8)


    def forward(self, x):
        x = self.attn1(self.conv1(x))
        x = self.attn2(self.conv2(x))
        x = self.attn3(self.conv3(x))
        x = self.attn4(self.conv4(x))
        x = self.attn5(self.conv5(x))
        x = self.attn6(self.conv6(x))
        x = self.convFinal(x)
        x = x.view(-1, self.nf*8)

        return x
    
class ATTN_TraceEncoder_2DCNN_encode_24576_64(torch.nn.Module):
    def __init__(self, input_len, dim):
        super(ATTN_TraceEncoder_2DCNN_encode_24576_64, self).__init__()

        self.dim = dim
        self.nf = 16
        self.nc = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.nc, 
                      out_channels = self.nf, 
                      kernel_size = (4,4), 
                      stride = (2, 2), 
                      padding = 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf, 
                      out_channels = self.nf, 
                      kernel_size = (4,4), 
                      stride = (2, 2), 
                      padding = 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf, 
                      out_channels = self.nf*2, 
                      kernel_size = (4,4), 
                      stride = 2, 
                      padding = 1),
            nn.BatchNorm2d(self.nf*2),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*2, 
                      out_channels = self.nf*4, 
                      kernel_size = (4,4), 
                      stride = 2, 
                      padding = 1),
            nn.BatchNorm2d(self.nf*4),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*4, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.nf*8, 
                      kernel_size = (4,3), 
                      stride = (2, 1), 
                      padding = (1, 1)),
            nn.BatchNorm2d(self.nf*8),
            nn.LeakyReLU(0.2, inplace=True))

        
        self.convFinal = nn.Sequential(
            nn.Conv2d(in_channels = self.nf*8, 
                      out_channels = self.dim, 
                      kernel_size = (6,4), 
                      stride = (1, 1), 
                      padding = 0),
            nn.Tanh())
        self.apply(weights_init)    
        
        self.attn1 = CBAM(self.nf)
        self.attn2 = CBAM(self.nf)
        self.attn3 = CBAM(self.nf * 2)
        self.attn4 = CBAM(self.nf * 4)
        self.attn5 = CBAM(self.nf * 8)
        self.attn6 = CBAM(self.nf * 8)
        self.attn7 = CBAM(self.nf * 8)
        self.attn8 = CBAM(self.nf * 8)
        self.attn9 = CBAM(self.nf * 8)
        self.attn10 = CBAM(self.nf * 8)
        self.attn11 = CBAM(self.nf * 8)
        self.attn12 = CBAM(self.nf * 8)


    def forward(self, x):
        x = self.attn1(self.conv1(x))
        x = self.attn2(self.conv2(x))
        x = self.attn3(self.conv3(x))
        x = self.attn4(self.conv4(x))
        x = self.attn5(self.conv5(x))
        x = self.attn6(self.conv6(x))
        x = self.attn7(self.conv7(x))
        x = self.attn8(self.conv8(x))
        x = self.attn9(self.conv9(x))
        x = self.attn10(self.conv10(x))
        x = self.attn11(self.conv11(x))
        x = self.attn12(self.conv12(x))
        x = self.convFinal(x)
        x = x.view(-1, self.nf*8)

        return x