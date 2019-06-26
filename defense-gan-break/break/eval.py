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

BIG_BATCH_SIZE = 10#100
BATCH_SIZE = 1
R = 10
L = 200
I = int(L*0.8)
LAM = 0.00025
CRIT = nn.CrossEntropyLoss(reduce=False)
LARGE_NUM = 10000
RANGE = ch.range(0,BIG_BATCH_SIZE*R-1).long()
SGD_LR = 10.
folder_list = ['./intermediates_2.5','./intermediates_3.0','./intermediates_3.5', './intermediates_3.0_200', './intermediates_3.0_300']
NUM_FOLDERS = len(folder_list)

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
    zhat = ch.randn(R*ims.shape[0], 128)
    targets = ims.repeat(R, 1)
    zhat.requires_grad_()
    opt = optim.SGD([zhat], lr=SGD_LR, momentum=0.7)
    lr_maker = StepLR(opt, step_size=I)

    for i in range(num_steps):
        opt.zero_grad()
        loss_mat = ((gan(zhat) - targets)**2).mean(-1)
        total_loss = loss_mat.clone()
        if (i+1) % 100 == 0:
            print("Iteration %d | Distance Loss %f" % (i, loss_mat.mean()))
        total_loss.mean().backward()
        opt.step()
        lr_maker.step()

    distance_mat = ch.stack(loss_mat.chunk(R, 0), 0) 
    image_mat = ch.stack(gan(zhat).chunk(R, 0), 0)
    zh_mat = ch.stack(zhat.chunk(R, 0), 0)
    ind = (-distance_mat).argmax(0)
    im_range = ch.range(0,ims.shape[0]-1).long()
    best_ims = image_mat[ind,im_range,:]
    best_zhs = zh_mat[ind,im_range,:]

    return best_ims.clone().detach(), best_zhs.clone().detach()

def accuracy(): 
    total_correct = 0.0
    total_incorrect_inbds = 0.0
    total = 0.0
    i = 0
    try:
        for ims_, targets in testloader:
            targets = targets.cuda()
            all_ints = []
            all_orig = []
            for j in range(BIG_BATCH_SIZE//BATCH_SIZE):
                for folder in folder_list: 
                    intermediates = ch.load("%s/batch_%d_attack" % (folder,i,))
                    originals = ch.load("%s/batch_%d_orig" % (folder,i,))
                    all_ints.append(intermediates)
                    all_orig.append(originals)
                i += 1
            intermediates = ch.cat(all_ints, 0)
            originals = ch.cat(all_orig, 0)
            images, _ = latent_space_opt(intermediates.view(-1, 784).cuda(), targets, num_steps=L)
            with ch.no_grad():
                out = cla(images.view(-1,1,28,28).cuda())
                
                preds = out.argmax(1)
                preds_repeat = ch.stack(preds.chunk(BIG_BATCH_SIZE),0)
                
                targets_repeat = ch.stack(targets.repeat(NUM_FOLDERS).chunk(NUM_FOLDERS),1)
                print(preds_repeat, targets_repeat)
                correct = preds_repeat == targets_repeat

                total_correct += ch.all(correct,1).float().sum()
                total += preds_repeat.shape[0]

                norms = (intermediates-originals.view(originals.shape[0], -1)).float().pow(2).sum(-1).pow(0.5)
                norms = ch.stack(norms.chunk(BIG_BATCH_SIZE),0)
                print(norms)
                print(norms.shape)
                total_incorrect_inbds += ch.any((1-correct)*(norms<4),1).float().sum()
                #total_correct += (preds == targets).float().sum()
                #norms = (intermediates-originals.view(BIG_BATCH_SIZE, -1)).float().pow(2).sum(-1).pow(0.5)
                #total_incorrect_inbds += ((1 - (preds == targets).float())*((norms < 4).float())).sum()
                #total += preds.shape[0]
            print(total_correct/total, total_incorrect_inbds/total)
    except:
        return total, total_correct/total, total_incorrect_inbds/total

total, a,b = accuracy()
print("Total %f | Adversarial accuracy %f | Adversarial in-bounds accuracy %f" % (total,a.item(), 1-b.item()))
