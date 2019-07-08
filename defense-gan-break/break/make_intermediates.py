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
R = 100 # defense-gan performs R restarts for inversion. we want to make our attack robust to this
L = 50000
I = int(L*0.8)

THR = 0.02 # this threshold is given by (784*0.0051)^2 / 784.
# Athalye et al use a per pixel l2 perturbation of 0.0051, while we want
# the average square L2 perturbation

CRIT = nn.CrossEntropyLoss(reduce=False)
LARGE_NUM = 10000
RANGE = ch.range(0,BATCH_SIZE*R-1).long()
SAMPLES_PER_ITER=100 # in the Expectation over Transformation part of the attack, we average over these many transformations
ROBUSTNESS_NORM=0.5 # we want our attack to be robust to noise introduced by Defense-GAN's inversion
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

def attack(ims, labels, num_steps=L, batch_num=None, ind=None):
    ims = ims.view(-1, IMAGE_DIM)
    # the inverses for images in the batches are precomputed and stored
    # in the folder encodings/ by the script encode.py
    assert os.path.isfile("encodings/saved_latents_%d" % (batch_num,))
    zh = ch.load("encodings/saved_latents_%d" % (batch_num,))[ind*BATCH_SIZE:(ind+1)*BATCH_SIZE,...]

    # create R copies for the inverse of each image. we want each of these to become adversarial attacks
    zhat = zh.repeat(R, 1)
    targets = ims.repeat(R, 1)
    zhat.requires_grad_()

    # create a mask which checks whether attacks are done/not done
    not_dones_mask = ch.ones(zhat.shape[0])
    # initialize the dual variable/lagrange multiplier for the perturbation constraint
    LAM = 1000*ch.ones_like(not_dones_mask)
    LAM.requires_grad_()

    opt = optim.Adam([zhat], lr=ADAM_LR)
    lam_opt = optim.SGD([LAM], lr=10000.0)

    lr_maker = StepLR(opt, step_size=I)

    for i in range(num_steps):
        opt.zero_grad()

        # Image Recovery Loss
        gen = gan(zhat)
        # this computes the average square L2 perturbation for each restart of each image
        loss_mat = ((gen - targets)**2).mean(-1)
        
        # if the perturbation is below THR/2, don't include it in the loss, set it to some constant
        loss_mat = loss_mat*(loss_mat > THR/2).float() - (loss_mat <= THR/2).float()
        total_loss = loss_mat.clone()
        ttf = targets.view(R*BATCH_SIZE,1,28,28)
        gtf = gen.view(ttf.shape)
        loss_extra = 0

        # Min-max CW loss
        # EOT part of our attack
        for j in range(SAMPLES_PER_ITER):
            # sample random noise of shape (R*BATCH_SIZE, 1, 28, 28)
            r = ch.randn_like(gtf)
            # renormalize the noise to unit norm
            norm_r = ch.sqrt(r.view(-1, IMAGE_DIM).pow(2).sum(-1)).view(-1, 1, 1, 1)
            # add the scaled noise to the output of the generator
            cla_res = cla.main(gtf + ROBUSTNESS_NORM*r/norm_r)

            # we want to compute the largest logit among all classes which are NOT the true label
            cla_res_second_best = cla_res.clone()
            cla_res_second_best[:,labels.repeat(R)] = -LARGE_NUM
            pred_classes = cla_res_second_best.argmax(-1)
            # CW loss is the logit for true label minus largest logit which is NOT the true label 
            loss_new = cla_res[RANGE,labels.repeat(R)] - cla_res[RANGE,pred_classes]
            # loss_extra was initialized to zero.
            # for each iteration of the variable j, we add a new noise vector to each restart, and
            # compute the aggregated loss over all noise vectors. 
            # this forms the expectation over transformation attack
            loss_extra += loss_new

        # average over the number of noise vectors we added to each restart
        loss_extra = loss_extra/SAMPLES_PER_ITER
        # add the CW loss to perturbation loss*lagrange multiplier
        total_loss = loss_extra.mean() + total_loss * LAM
        #new_loss = ch.log(ch.exp(loss_extra).sum())*LAM
        #total_loss += new_loss

        if i % 50 == 0:
            print("Iteration %d | Distance Loss %f | Adversarial Loss %f" % (i, loss_mat.mean(), loss_extra.mean()))

        # split the matrices into shape (R, BATCH_SIZE, 1, 28, 28)
        # this way all the restarts for each image are along the 0th dimension
        cla_mat = ch.stack(loss_extra.chunk(R, 0), 0)
        distance_mat = ch.stack(loss_mat.chunk(R, 0), 0) 
        # the output of a restart is done if it is within the perturbation budget,
        # and the classifier loss is sufficiently negative
        not_dones_mask = 1 - (distance_mat <= THR).float()*(cla_mat <= -1).float()
        # check if ALL the restarts of an image are done, and repeat it R times
        not_dones_mask = not_dones_mask.min(dim=0)[0].repeat(R)
        not_dones_mask = not_dones_mask.view(-1, 1)

        image_mat = ch.stack(gan(zhat).chunk(R, 0), 0)
        im_range = ch.range(0,BATCH_SIZE-1).long()

        ind = (-cla_mat - LARGE_NUM*(distance_mat > THR).float()).argmax(0) # Pick argmin of cla_mat, that is within the perturbation budget
        loss_at_best = cla_mat[ind,im_range]
        dists_at_best = distance_mat[ind,im_range]

        # if at least 90% of the restarts are adversarial, then we are done
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

        # this is the loss for the primal variable z
        ((total_loss*not_dones_mask).mean()/not_dones_mask.mean()).backward(retain_graph=True)
        opt.step()

        # Lambda step
        lam_opt.zero_grad()
        # this is the loss for the lagrange multiplier/dual variable lambda
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
            # create an attack for the j^th image in the k^th batch
            intermediates, _ = attack(ims.view(-1, IMAGE_DIM).cuda(), targets, num_steps=10000, batch_num=k, ind=j)
            # i keeps track of the total number of images seen so far
            ch.save(intermediates, "intermediates/batch_%d_attack" % (i,))
            ch.save(ims, "intermediates/batch_%d_orig" % (i,))
            i += 1

make_intermediates()
