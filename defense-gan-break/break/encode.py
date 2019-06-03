import torch as ch
import os.path
from torch import optim
from torch import nn
from torch.utils import data
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models import Classifier

ch.set_default_tensor_type('torch.cuda.FloatTensor')

BATCH_SIZE = 50
R = 20
L = 50000
I = int(L*0.8)
LAM = 0.00025
CRIT = nn.CrossEntropyLoss(reduce=False)
LARGE_NUM = 10000
RANGE = ch.range(0,BATCH_SIZE*R-1).long()
SAMPLES_PER_ITER=100

gan = nn.DataParallel(ch.load("ckpts/netG"))
for p in gan.parameters():
    p.requires_grad_(False)
cla = ch.load("ckpts/classifier")
for p in cla.parameters():
    p.requires_grad_(False)

testloader = data.DataLoader(
	datasets.MNIST('../data', download=True, train=False, transform=transforms.Compose([
		   transforms.ToTensor(),
	   ])), batch_size=BATCH_SIZE, shuffle=False)

def latent_space_opt(ims, labels, num_steps=L):
    ims = ims.view(-1, 784)
    zhat = ch.randn(R*BATCH_SIZE, 128)
    targets = ims.repeat(R, 1)
    zhat.requires_grad_()
    opt = optim.SGD([zhat], lr=10.0, momentum=0.7)
    lr_maker = StepLR(opt, step_size=I)

    for i in range(num_steps):
        opt.zero_grad()
        loss_mat = ((gan(zhat) - targets)**2).mean(-1)
        total_loss = loss_mat.clone()
        if i % 500 == 0:
            print("Iteration %d | Distance Loss %f" % (i, loss_mat.mean()))
        total_loss.mean().backward()
        opt.step()
        lr_maker.step()

    distance_mat = ch.stack(loss_mat.chunk(R, 0), 0) 
    image_mat = ch.stack(gan(zhat).chunk(R, 0), 0)
    zh_mat = ch.stack(zhat.chunk(R, 0), 0)
    ind = (-distance_mat).argmax(0)
    im_range = ch.range(0,BATCH_SIZE-1).long()
    best_ims = image_mat[ind,im_range,:]
    best_zhs = zh_mat[ind,im_range,:]

    return best_ims.clone().detach(), best_zhs.clone().detach()

def encode(ims, labels, num_steps=L):
    ims = ims.view(-1, 784)
    fname = "encodings/saved_latents_%d" % (num_steps,)
    if os.path.isfile(fname):
        print("Already saved this one!")
    else:
        _, zh = latent_space_opt(ims, labels)
        ch.save(zh, fname)
    return None, None

def encode_all(): 
    i = 0
    for ims, targets in testloader:
        ims, targets = ims.cuda(), targets.cuda()
        intermediates, _ = encode(ims.view(-1, 784).cuda(), targets, num_steps=i)
        i += 1

encode_all()
