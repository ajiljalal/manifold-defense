import os, sys
import numpy as np
import torch
from torch.utils import data
import torch.autograd as autograd
import torch.optim as optim
from models import Discriminator, Generator
from torchvision import datasets, transforms

sys.path.append(os.getcwd())
torch.manual_seed(1)

BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
trainloader = data.DataLoader(
	datasets.MNIST('../data', download=True, train=True, transform=transforms.Compose([
		   transforms.ToTensor(),
	   ])), batch_size=BATCH_SIZE, shuffle=True)
def train_loader_gen():
    while True:
        for images, labels in trainloader:
            yield images.view(-1, 784)

train_loader = train_loader_gen()

train_loader = iter(train_loader)
test_loader = data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.Compose([
		   transforms.ToTensor(),
	   ])), batch_size=BATCH_SIZE, shuffle=True)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size()).cuda()
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

netG = Generator().cuda()
netD = Discriminator().cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1]).cuda()
mone = one * -1

for iteration in range(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):
        _data = next(train_loader)
        real_data = torch.Tensor(_data).cuda()
        real_data_v = autograd.Variable(real_data)
        netD.zero_grad()

        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128).cuda()
        noisev = autograd.Variable(noise)  # totally freeze netG
            
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()
        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128).cuda()
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 100 == 99:
        dev_disc_costs = []
        for images,_ in test_loader:
            imgs = torch.Tensor(images).cuda()
            imgs_v = autograd.Variable(imgs)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        print("Iteration %d | Disc cost: %f" % (iteration, np.mean(dev_disc_costs)))
        torch.save(netG, "ckpts/netG")
