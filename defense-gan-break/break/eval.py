import torch as ch
import os.path
from torch import optim
from torch import nn
from torch.utils import data
import torch.autograd as autograd
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from models import Classifier

ch.set_default_tensor_type('torch.cuda.FloatTensor')

BIG_BATCH_SIZE = 100
BATCH_SIZE = 1
R = 20
L = 50000
I = int(L*0.8)
LAM = 0.00025
CRIT = nn.CrossEntropyLoss(reduce=False)
LARGE_NUM = 10000
RANGE = ch.range(0,BIG_BATCH_SIZE*R-1).long()
SAMPLES_PER_ITER=100

gan = nn.DataParallel(ch.load("ckpts/netG")).cuda()
for p in gan.parameters():
    p.requires_grad_(False)
cla = nn.DataParallel(ch.load("ckpts/classifier")).cuda()
for p in cla.parameters():
    p.requires_grad_(False)

testloader = data.DataLoader(
	datasets.MNIST('../data', download=True, train=False, transform=transforms.Compose([
		   transforms.ToTensor(),
	   ])), batch_size=BIG_BATCH_SIZE, shuffle=False)

def latent_space_opt(ims, labels, num_steps=L):
    ims = ims.view(-1, 784)
    zhat = ch.randn(R*BIG_BATCH_SIZE, 128)
    targets = ims.repeat(R, 1)
    zhat.requires_grad_()
    opt = optim.SGD([zhat], lr=500.0, momentum=0.7)
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
    im_range = ch.range(0,BIG_BATCH_SIZE-1).long()
    best_ims = image_mat[ind,im_range,:]
    best_zhs = zh_mat[ind,im_range,:]

    return best_ims.clone().detach(), best_zhs.clone().detach()

def accuracy(): 
    total_correct = 0.0
    total_correct_inbds = 0.0
    total = 0.0
    i = 0
    try:
        for ims_, targets in testloader:
            targets = targets.cuda()
            all_ints = []
            all_orig = []
            for j in range(BIG_BATCH_SIZE//BATCH_SIZE):
                intermediates = ch.load("intermediates/batch_%d_attack" % (i,))
                originals = ch.load("intermediates/batch_%d_orig" % (i,))
                all_ints.append(intermediates)
                all_orig.append(originals)
                i += 1
            intermediates = ch.cat(all_ints, 0)
            originals = ch.cat(all_orig, 0)
            images, _ = latent_space_opt(intermediates.view(-1, 784).cuda(), targets, num_steps=20000)
            with ch.no_grad():
                out = cla(images.view(-1,1,28,28).cuda())
                preds = out.argmax(1)
                total_correct += (preds == targets).float().sum()
                norms = (intermediates-originals.view(BIG_BATCH_SIZE, -1)).float().pow(2).sum(-1).pow(0.5)
                total_correct_inbds += ((1 - (preds == targets).float())*((norms < 4).float())).sum()
                total += preds.shape[0]
    except:
        return total_correct/total, total_correct_inbds/total

a,b = accuracy()
print("Adversarial accuracy %f | Adversarial in-bounds accuracy %f" % (a.item(), 1-b.item()))
