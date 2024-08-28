import torch.nn as nn

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
            # state size. (ngf) x 6
            # 4 x 64
            nn.ConvTranspose2d(self.dim, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        output = self.main(x)
        return output



