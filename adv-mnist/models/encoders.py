import torch as ch
import itertools
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as snorm
import numpy as np

class IdentityEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args,**kwargs):
        return x

'''
class MNISTVAE(nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True):
        super(MNISTVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, embed_feats)
        self.fc22 = nn.Linear(400, embed_feats)
        self.fc3 = nn.Linear(embed_feats, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500,784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = ch.exp(0.5*logvar)
        eps = ch.randn_like(std)
        return mu + eps*std

    def decode(self, z,square=True):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        if square:
            return ch.sigmoid(self.fc5(h4)).view(-1,1,28,28)
        else:
            return ch.sigmoid(self.fc5(h4))

    def forward(self, x, latent=False, square=True):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        if latent:
            return self.decode(z,square), mu, logvar
        else:
            return self.decode(z,square)
'''
'''
class MNISTVAE(nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True):
        super(MNISTVAE, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding=0, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.fc_mu = nn.Linear(num_feats*8, embed_feats)
        self.fc_sigma = nn.Linear(num_feats*8, embed_feats)
        
        self.fc_decode_1 = nn.Linear(embed_feats, 3136)
        self.deconv_decode_1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.deconv_decode_2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_decode_1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, stride=1)


    def encode(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.shape[0], -1)
        return self.fc_mu(out), self.fc_sigma(out)
        

    def reparameterize(self, mu, logvar):
        std = ch.exp(0.5*logvar)
        eps = ch.randn_like(std)
        return mu + eps*std

    def decode(self, z,square=True):
        out = F.relu(self.fc_decode_1(z))
        out = out.view(out.shape[0], 16, 14, 14)
        out = F.relu(self.deconv_decode_1(out))
        out = F.relu(self.deconv_decode_2(out))
        out = self.conv_decode_1(out)
        if square:
            return ch.sigmoid(out).view(-1,1,28,28)
        else:
            return ch.sigmoid(out)

    def forward(self, x, latent=False, square=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if latent:
            return self.decode(z,square), mu, logvar
        else:
            return self.decode(z,square)
'''

class MNISTVAE(nn.Module):
    def __init__(self, embed_feats, leaky_relu=True):
        super(MNISTVAE, self).__init__()

        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, 20)
        self.fc22 = nn.Linear(500, 20)
        self.fc3 = nn.Linear(20,500)#, bias=False)
        self.fc4 = nn.Linear(500,500)
        self.fc5 = nn.Linear(500, 784)#, bias=False)
        
        self.leaky_relu = leaky_relu
        if leaky_relu:
            self.nonlinearity = nn.LeakyReLU(0.1)
        else:
            self.nonlinearity = nn.ReLU()

    def encode(self, x):
        h1 = self.nonlinearity(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = ch.exp(0.5*logvar)
        eps = ch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z,square=True):
        out = self.nonlinearity(self.fc3(z))
        out = self.nonlinearity(self.fc4(out))
        if square:
            return ch.sigmoid(self.fc5(out)).view(-1,1,28,28)
        else:
            return ch.sigmoid(self.fc5(out))

    def forward(self, x,latent=False):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        if latent:
             return self.decode(z), mu, logvar
        else:
             return self.decode(z)

