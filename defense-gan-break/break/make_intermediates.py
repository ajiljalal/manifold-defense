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

BIG_BATCH_SIZE = 50
BATCH_SIZE = 1
R = 100
L = 50000
I = int(L*0.8)


THR = 0.02
CRIT = nn.CrossEntropyLoss(reduce=False)
LARGE_NUM = 10000
RANGE = ch.range(0,BATCH_SIZE*R-1).long()
SAMPLES_PER_ITER=100
ROBUSTNESS_NORM=0.5
ADAM_LR = 0.05
SGD_LR = 10.0
IMAGE_DIM = 784
LATENT_DIM = 28

gan = nn.DataParallel(ch.load("ckpts/netG")).cuda()
for p in gan.parameters():
    p.requires_grad_(False)
cla = ch.load("ckpts/classifier").cuda()
for p in cla.parameters():
    p.requires_grad_(False)

testloader = data.DataLoader(
	datasets.MNIST('../data', download=True, train=False, transform=transforms.Compose([
		   transforms.ToTensor(),
	   ])), batch_size=BIG_BATCH_SIZE, shuffle=False)

def latent_space_opt(ims, labels, num_steps=L):
    ims = ims.view(-1, IMAGE_DIM)
    zhat = ch.randn(R*BATCH_SIZE, LATENT_DIM)
    targets = ims.repeat(R, 1)
    zhat.requires_grad_()
    opt = optim.SGD([zhat], lr=SGD_LR, momentum=0.7)
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

def attack(ims, labels, num_steps=L, batch_num=None, ind=None):
    ims = ims.view(-1, IMAGE_DIM)
    assert os.path.isfile("encodings/saved_latents_%d" % (batch_num,))
    zh = ch.load("encodings/saved_latents_%d" % (batch_num,))[ind*BATCH_SIZE:(ind+1)*BATCH_SIZE,...]

    zhat = zh.repeat(R, 1)
    targets = ims.repeat(R, 1)
    zhat.requires_grad_()

    not_dones_mask = ch.ones(zhat.shape[0])
    LAM = 1000*ch.ones_like(not_dones_mask)
    LAM.requires_grad_()

    opt = optim.Adam([zhat], lr=ADAM_LR)
    lam_opt = optim.SGD([LAM], lr=10000.0)

    lr_maker = StepLR(opt, step_size=I)

    for i in range(num_steps):
        opt.zero_grad()

        # Image Recovery Loss
        gen = gan(zhat)
        loss_mat = ((gen - targets)**2).mean(-1)
        loss_mat = loss_mat*(loss_mat > THR/2).float() - (loss_mat <= THR/2).float()
        total_loss = loss_mat.clone()
        ttf = targets.view(R*BATCH_SIZE,1,28,28)
        gtf = gen.view(ttf.shape)
        loss_extra = 0

        # Min-max CW loss
        for j in range(SAMPLES_PER_ITER):
            r = ch.randn_like(gtf)
            norm_r = ch.sqrt(r.view(-1, IMAGE_DIM).pow(2).sum(-1)).view(-1, 1, 1, 1)
            cla_res = cla.main(gtf + ROBUSTNESS_NORM*r/norm_r)

            cla_res_second_best = cla_res.clone()
            cla_res_second_best[:,labels.repeat(R)] = -LARGE_NUM
            true_classes = cla_res_second_best.argmax(-1)
            loss_new = cla_res[RANGE,labels.repeat(R)] - cla_res[RANGE,true_classes]
            loss_extra += loss_new

        loss_extra = loss_extra/SAMPLES_PER_ITER
        total_loss = loss_extra.mean() + total_loss * LAM
        #new_loss = ch.log(ch.exp(loss_extra).sum())*LAM
        #total_loss += new_loss

        if i % 50 == 0:
            print("Iteration %d | Distance Loss %f | Adversarial Loss %f" % (i, loss_mat.mean(), loss_extra.mean()))

        cla_mat = ch.stack(loss_extra.chunk(R, 0), 0)
        distance_mat = ch.stack(loss_mat.chunk(R, 0), 0) 
        not_dones_mask = 1 - (distance_mat <= THR).float()*(cla_mat <= -1).float()
        not_dones_mask = not_dones_mask.min(dim=0)[0].repeat(R)
        not_dones_mask = not_dones_mask.view(-1, 1)

        image_mat = ch.stack(gan(zhat).chunk(R, 0), 0)
        im_range = ch.range(0,BATCH_SIZE-1).long()

        ind = (-cla_mat - LARGE_NUM*(distance_mat > THR).float()).argmax(0) # Pick argmin of cla_mat 
        loss_at_best = cla_mat[ind,im_range]
        dists_at_best = distance_mat[ind,im_range]

        if not_dones_mask.mean() < 0.1 or i == num_steps - 1:
            zh_mat = ch.stack(zhat.chunk(R, 0), 0)
            best_ims = image_mat[ind,im_range,:]
            best_zhs = zh_mat[ind,im_range,:]
            return best_ims.clone().detach(), zhat.clone().detach()
        elif i % 1 == 0:
            print("----")
            print("Norms", dists_at_best)
            print("Losses", loss_at_best)
            #print("----")
            #print("Maximum loss (of best images)", loss_at_best.max())
            #print("Mean loss (of best images)", loss_at_best.mean())
            #print("----")
            print("Success rate: ", not_dones_mask.mean())
            print("Lambda: ", LAM)

        ((total_loss*not_dones_mask).mean()/not_dones_mask.mean()).backward(retain_graph=True)
        opt.step()

        # Lambda step
        lam_opt.zero_grad()
        (-(total_loss*not_dones_mask).mean()/not_dones_mask.mean()).backward()
        lam_opt.step()
        #LAM.data = ch.max(LAM, 0)[0]

        lr_maker.step()

def make_intermediates(): 
    i = 0
    for k, (ims_, targets_) in enumerate(testloader):
        for j in range(BIG_BATCH_SIZE//BATCH_SIZE):
            ims, targets = ims_[j*BATCH_SIZE:(j+1)*BATCH_SIZE,...].cuda(), \
                            targets_[j*BATCH_SIZE:(j+1)*BATCH_SIZE,...].cuda()
            intermediates, _ = attack(ims.view(-1, IMAGE_DIM).cuda(), targets, num_steps=10000, batch_num=k, ind=j)
            ch.save(intermediates, "intermediates/batch_%d_attack" % (i,))
            ch.save(ims, "intermediates/batch_%d_orig" % (i,))
            i += 1

make_intermediates()
