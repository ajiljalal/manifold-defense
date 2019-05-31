import torch as ch
import torch.nn.functional as F
import torch.optim as optim # Optimizers
import sys
from torchvision import transforms
from attacks import pgd_l2, pgd_linf, opmaxmin, ce
from argparse import ArgumentParser
from models import resnet
import numpy as np
from YellowFin_Pytorch.tuner_utils.yellowfin import YFOptimizer

parser = ArgumentParser()
parser.add_argument('--dataset', choices=["mnist", "cifar"], required=True,
        help="Which dataset to use for training")
parser.add_argument('--mode', type=str, choices=['linf', 'l2'], 
        help="Perturbation model")
parser.add_argument('--eps', type=float, 
        help="Adversarial perturbation budget")
parser.add_argument('--num-pgd', type=int, default=10,
        help="Number of steps to use while making adversarial examples for training")
parser.add_argument('--pgd-lr', type=float, 
        help="Learning rate for PGD")
parser.add_argument('--validation-set', action='store_true',
        help="If given, use a validation set")
parser.add_argument('--opt', default="sgd", choices=["adam", "sgd", 'yf'])
parser.add_argument('--sgd-lr', type=float, default=1e-2,
        help="SGD learning rate")
parser.add_argument('--no-norm', action='store_true',
        help="Should not normalize the image to the proper imagenet stats")
parser.add_argument('--save-str', type=str, default="",
        help="A unique identifier to save with")
parser.add_argument('--resume', type=str)
parser.add_argument('--num-epochs', default=500, type=int,
        help="Number of epochs to train for")
parser.add_argument('--save-iters', default=10, type=int,
        help="Save after these many epochs")
parser.add_argument('--dataset-size', default=None, type=int,
        help="Number of samples to use")
parser.add_argument('--random-step', action='store_true', 
        help="Whether to start by taking a random step in PGD attack")
parser.add_argument('--op-attack', action='store_true', 
        help="If given, perform overpowered attack")
parser.add_argument('--op-eps', type=float, default=None,
        help="Perturbation budget for the overpowered attack, defaults to regular budget")
parser.add_argument('--op-generator', type=str,
        help="Path to generator network for overpowered attack")
parser.add_argument('--op-embed-feats', type=int, default=20,
        help="Dimension of latent space of the generator")
parser.add_argument('--op-iter', type=int, default=5,
        help='after how many epochs should the overpowered attack be run')
parser.add_argument('--op-weight', type=float, default=1e-2,
        help="weighting for loss function on overpowered attack samples")

args = parser.parse_args()


MODE = args.mode
IMAGE_DIM = 32*32*3 if args.dataset == "cifar" else 784

if args.dataset == "cifar":
    if not args.validation_set:
        from cifar_config import no_norm_fullloader as trainloader
        from cifar_config import no_norm_testloader as testloader
    else:
        from cifar_config import no_norm_trainloader as trainloader
        from cifar_config import no_norm_testloader as testloader
        from cifar_config import no_norm_validationloader as validationloader
    from models.resnet import ResNet18 as Classifier
else:
    from models.encoders import MNISTVAE as Generator
    from models.simple_models import MNISTClassifier as Classifier
    if not args.validation_set:
        from mnist_config import testloader
        from mnist_config import trainloaderfull as trainloader
    else:
        from mnist_config import trainloader, testloader, validationloader
    
    
if args.op_attack:
    generator = Generator(args.op_embed_feats).cuda()
    generator.load_state_dict(ch.load(args.op_generator))
    generator.eval()
    if args.op_eps is None:
        args.op_eps = args.eps

net = Classifier().cuda()
if args.resume:
    net_dict = ch.load("results/%s_%s_%s" % (args.dataset, MODE, args.resume))
    net.load_state_dict(net_dict)

NUM_STEPS = args.num_pgd
DEFAULT_EPS = {
    "l2": 0.1,
    "linf": 8/255.0,
}
EPS = args.eps if args.eps is not None else DEFAULT_EPS[args.mode]
LR = args.pgd_lr if args.pgd_lr is not None else 2*EPS/NUM_STEPS

NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

SAVE_ITERS = args.save_iters
ATTACKS = {
    "l2": pgd_l2,
    "linf": pgd_linf,
}
attack = ATTACKS[MODE]


loss_fn = ch.nn.CrossEntropyLoss()
param_set = net.parameters()

if args.opt == "sgd":
    opt = optim.SGD(param_set, lr=args.sgd_lr, momentum=0.9, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[50,100,150,500], gamma=0.1)
elif args.opt == "adam":
    opt = optim.Adam(param_set, lr=args.sgd_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.num_epochs+1], gamma=0.1)
elif args.opt == 'yf':
    opt = YFOptimizer(param_set, lr=args.sgd_lr, clip_thresh=None, adapt_clip=False)
    #scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.num_epochs+1], gamma=0.1)


def se(x1, x2, reduction='mean'):
    y = ch.norm((x1-x2).view(x1.shape[0],-1),dim=-1,p=2)**2
    if reduction=='sum':
        return ch.sum(y)
    elif reduction=='mean':
        return ch.mean(y)
    else: 
        return y

best_adv_acc = 0.
for ep in range(1,args.num_epochs+1):
    if args.op_attack and (ep-1) % args.op_iter ==0:
        net.train()
        opt.zero_grad()
        batch1, batch2, is_adv = opmaxmin(net,generator.decode,args.op_eps,num_images=50,\
                num_steps=500,embed_feats=args.op_embed_feats,z_lr=1e-4,lambda_lr=1e-4,ind=ep)
        if ch.sum(is_adv) > 0:
            loss = args.op_weight*ch.sum(ce(net(batch1), net(batch2),reduction=None)*is_adv)/ch.sum(is_adv)
            loss.backward()
            opt.step()
        else:
            pass
    
    total_ims_seen = 0
    val_num_correct = 0
    val_num_total = 0
    for i, (images, labels) in enumerate(trainloader):

        if args.dataset_size is not None and total_ims_seen > args.dataset_size:
            break
        net.train()
        # Shape of images: (BATCH_SIZE x channels x width x height)
        # Shape of labels: (BATCH_SIZE)
        images, labels = images.cuda(), labels.cuda()
        _args = [net, images, labels, NUM_STEPS, LR, EPS, args.random_step]
        new_ims = attack(*_args).detach()
        opt.zero_grad()

        pred_probs = net(new_ims)
        loss = loss_fn(pred_probs, labels)
        pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
        acc = (pred_classes == labels).float().mean()
        if (i+1) % 100 == 0:
            print("Epoch {0} | Iteration {1} | Loss {2} | Adv Acc {3}".format(ep, i+1, loss, acc))
        loss.backward()
        opt.step()
        total_ims_seen += images.shape[0]

    net.eval()
    if args.validation_set:
        for i, (images, labels) in enumerate(validationloader):
            net.eval()
            # Shape of images: (BATCH_SIZE x channels x width x height)
            # Shape of labels: (BATCH_SIZE)
            images, labels = images.cuda(), labels.cuda()
            _args = [net, images, labels, NUM_STEPS, LR, EPS, args.random_step]
            new_ims = attack(*_args).detach()
            pred_probs = net(new_ims)
            loss = loss_fn(pred_probs, labels)
            pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)

            val_num_correct += (pred_classes == labels).float().sum()
            val_num_total += labels.shape[0]
        

        print("###### EPOCH {0} COMPLETE ######".format(ep))
        print("Adversarial Validation Accuracy: %f" % (val_num_correct/val_num_total).cpu().item())
        print("############################")
    
    if val_num_correct/val_num_total > best_adv_acc:
        ch.save(net.state_dict(), "results/%s_%s_%s_best" % (args.dataset, MODE, args.save_str))
        print("Saved model...")
        best_adv_acc = val_num_correct/val_num_total
    scheduler.step()

    net.eval()
    with ch.no_grad():
        num_correct = 0
        num_total = 0
        for (images, labels) in testloader:
            images, labels = images.cuda(), labels.cuda()
            pred_probs = net(images) # Shape: (BATCH_SIZE x 10)
            pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
            num_correct += (pred_classes == labels).float().sum()
            num_total += labels.shape[0]
        print("###### EPOCH {0} COMPLETE ######".format(ep))
        print("Test Accuracy: %f" % (num_correct/num_total).cpu().item())
        print("############################")
    
    num_correct = 0
    num_total = 0
    net.eval()
    for (images, labels) in testloader:
        images, labels = images.cuda(), labels.cuda()
        _args = [net, images, labels, NUM_STEPS, LR, EPS, False]
        images = attack(*_args).detach()
        pred_probs = net(images) # Shape: (BATCH_SIZE x 10)
        pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
        num_correct += (pred_classes == labels).float().sum()
        num_total += labels.shape[0]
    print("###### EPOCH {0} COMPLETE ######".format(ep))
    print("Adversarial Test Accuracy: %f" % (num_correct/num_total).cpu().item())
    print("############################")
    
    if ep % SAVE_ITERS == 0:
        ch.save(net.state_dict(), "results/%s_%s_%s_%d" % (args.dataset, MODE, args.save_str, ep))
        print("Saved model...")
   
