import torch as ch
from YellowFin_Pytorch.tuner_utils.yellowfin import YFOptimizer
import numpy as np

loss_fn = ch.nn.CrossEntropyLoss()

# redefine CrossEntropyLoss so that it is differentiable
# wrt both arguments
def ce(x,y, reduction='mean'):
    softmax = ch.nn.Softmax(dim=-1)
    logsoftmax = ch.nn.LogSoftmax(dim=-1)

    ce = -1.*ch.sum(softmax(x)*logsoftmax(y),dim=-1) 
    if reduction=='mean':
        return ch.mean(ce)
    elif reduction=='sum':
        return ch.sum(ce)
    elif reduction==None:
        return ch.sum(ce)


def normed(t, new_shape):
    return t/ch.norm(t.view(t.shape[0], -1), dim=-1).view(new_shape)

def l2_projection(new_images, orig_images, eps):
    batch_size = new_images.shape[0]
    new_images_flat = new_images.view(batch_size, -1)
    orig_images_flat = orig_images.view(batch_size, -1)
    diff = new_images_flat - orig_images_flat
    diff_norms = ch.norm(diff, dim=-1, keepdim=True)
    # previous version can give division by zero
    clipped_diffs = ch.where(diff_norms<=eps, diff, eps*diff/diff_norms)
    #clip_mask = (diff_norms <= eps).float()
    #clipped_diffs = diff*clip_mask + eps * diff/(diff_norms) * (1-clip_mask)
    #print(clipped_diffs)
    clipped_ims = orig_images_flat + clipped_diffs
    return clipped_ims.view(orig_images.shape)

def linf_projection(new_images, orig_images, eps):
    return orig_images + ch.clamp(new_images - orig_images, -eps, eps)

# Generate PGD examples
def pgd_generic(new_net, orig_ims, correct_class, num_steps, lr, eps, use_noise, mode):
    new_shape = [-1, 1, 1, 1]
    projections = {
        "l2": l2_projection,
        "linf": linf_projection
    }
    proj = projections[mode]
    attack_ims = orig_ims.clone().detach()
    attack_ims.requires_grad = True
    if use_noise:
        if mode == 'linf':
            attack_ims = attack_ims + (ch.rand_like(orig_ims)*2 - 1) * eps
        else:
            scale = ch.Tensor([0.1])*eps# ch.rand(size=()) * eps
            noise = ch.randn_like(orig_ims)
            attack_ims = attack_ims + normed(noise, new_shape) * scale.cuda()
    for _ in range(num_steps):
        loss = loss_fn(new_net(attack_ims.clone()), correct_class)
        g, = ch.autograd.grad(loss, attack_ims)
        orig_shape = g.shape
        if mode == 'linf':
            g = ch.sign(g)
        elif mode == 'l2':
            g = g/(ch.norm(1e-16+g.view(g.shape[0], -1), dim=1, keepdim=True).view(*new_shape))
        attack_ims = attack_ims + lr * g
        attack_ims = proj(attack_ims, orig_ims, eps)
        attack_ims = ch.clamp(attack_ims, 0, 1)
    return attack_ims

# Generate PGD examples
def pgd_linf(*args):
    return pgd_generic(*args, mode="linf")

# Generate PGD examples
def pgd_l2(*args):
    return pgd_generic(*args, mode="l2")


def opmaxmin(cla, gan, eps, im_size=784, embed_feats=256, num_images=50, z_lr=5e-3, lambda_lr=1e-4,num_steps=1000, batch_num=None, ind=None):
    
    softmax = ch.nn.Softmax()
    logsoftmax = ch.nn.LogSoftmax()
    
    BATCH_SIZE = 1

    batch1 = ch.zeros((num_images, 1,28,28)).cuda()
    batch2 = ch.zeros((num_images, 1,28,28)).cuda()
    is_valid = ch.zeros(num_images).cuda()
    count = 0
    EPS = eps
    for i in range(num_images//BATCH_SIZE):

        z1 = ch.Tensor(ch.rand(BATCH_SIZE,embed_feats)).cuda() 
        z1.requires_grad = True
        z2 = ch.Tensor(ch.rand(z1.shape)).cuda()
        z2.requires_grad_()
        
        ones = ch.ones(z1.shape[0]).cuda()
        
        lambda_ = 1e0*ch.ones(z1.shape[0],1).cuda()
        lambda_.requires_grad = True

        opt1 = YFOptimizer([{'params':z1},{'params':z2}], lr=z_lr, clip_thresh=None, adapt_clip=False)
        opt2 = YFOptimizer([{'params':lambda_}], lr=lambda_lr, clip_thresh=None, adapt_clip=False)

        for j in range(num_steps):
                
            x1 = gan(z1)
            x2 = gan(z2)
            distance_mat = ch.norm((x1-x2).view(x1.shape[0],-1),dim=-1,keepdim=False) - EPS*ones
            
            cla_res1 = cla(x1).argmax(dim=-1)
            cla_res2 = cla(x2).argmax(dim=-1)
            
            #print('Cross entropy:%f \t distance=%f \t lambda=%f'%(ce(cla(x1),cla(x2)),distance_mat,lambda_))

            is_adv = 1 - (cla_res1==cla_res2).float()
            is_feasible = (distance_mat<=0).float() 
            not_valid = 1- (is_adv*is_feasible)
            if ch.sum(is_adv*is_feasible) == BATCH_SIZE:
#                 ind = (ch.abs(cla_res1 - cla_res2)*is_valid*is_feasible_mat).argmax(0)
                batch1[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = x1
                batch2[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = x2
                is_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = 1.
                break
            
            opt1.zero_grad()
            loss1 = (-1.* ch.sum(ce(cla(gan(z1)),cla(gan(z2)),reduction=None)*not_valid) + \
                     ch.sum(lambda_ * distance_mat*not_valid) + 1e-4*ch.sum(ch.norm(z1,dim=-1)*not_valid) +\
                     1e-4*ch.sum(ch.norm(z2,dim=-1)*not_valid))/ch.sum(not_valid)
            
            
            loss1.backward(retain_graph=True)
            opt1.step()
            
            for k in range(1):
                opt2.zero_grad()
                loss2 = -1.*ch.mean(lambda_ * distance_mat*(not_valid)) 
                loss2.backward()
                opt2.step()
                #lambda_ = lambda_.clamp(1e-3,1e5)
        batch1[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = x1
        batch2[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = x2
        is_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = is_adv * is_feasible
    
    count = ch.sum(is_valid)
    print('number of adversarial pairs found:%d\n'%(count))

    return batch1.detach(), batch2.detach(), is_valid


