import torch 
import torch.nn as nn


class img_classifier(nn.Module):
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
        self.apply(weights_init_classifier)

    def forward(self, x):
        x = x.view(-1, self.dim)
        out = self.main(x)
        return out
    
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        m.bias.data.fill_(0)


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
        self.apply(weights_init_classifier)

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

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
    

class CLF_D(nn.Module):
    def __init__(self, dim=128):
        super(CLF_D, self).__init__()

        self.fc1 = nn.Linear(dim, 100)
        self.activate1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)
        self.activate2 = nn.Sigmoid()
        
        self.apply(weights_init_classifier)

    def forward(self, x):
        x = self.activate1(self.fc1(x))
        x = self.activate2(self.fc2(x))
        
        return x
    
class CLF_C(torch.nn.Module):
    def __init__(self, dim=128, CLF_NUM=1):
        super(CLF_C, self).__init__()

        self.fc1 = nn.Linear(128, 100)
        self.activate1 = nn.ReLU()
        self.fc2 = nn.Linear(100, CLF_NUM)
        self.activate2 = nn.Softmax(dim = 1)
        
        self.apply(weights_init_classifier)

    def forward(self, x):
        x = self.activate1(self.fc1(x))
        x = self.activate2(self.fc2(x))
        
        return x