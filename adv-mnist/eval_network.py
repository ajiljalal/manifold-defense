import torch as ch
import torch.nn.functional as F
import torch.optim as optim # Optimizers
import sys
from models import resnet
from torchvision import transforms
from attacks import pgd_l2, pgd_linf
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar"],
        help="What dataset to evaluate against")
parser.add_argument("--load-str", type=str, required=True,
        help="String used to save model")
parser.add_argument("--load-epoch", type=int, default=None,
        help="Epoch to load net from. if None, use the model which has best val robustness")
parser.add_argument('--mode', type=str, choices=['linf', 'l2'], 
        help="Perturbation model")
parser.add_argument("--num-steps", type=int, default=100,
        help="Number of steps of PGD to run")
parser.add_argument("--eps", type=float, required=True,
        help="Adversarial perturbation budget")
parser.add_argument("--pgd-lr", type=float,
        help="Step size for PGD attack")
parser.add_argument('--random-step', action='store_true', default=False,
        help="Whether to start by taking a random step in PGD attack")
args = parser.parse_args()

from models.simple_models import MNISTClassifier as Model

if args.dataset == "cifar":
    from cifar_config import no_norm_testloader as testloader
else:
    from mnist_config import testloader


net = Model().cuda()
net_path = 'results/{0}_{1}_{2}_'.format(args.dataset, args.mode, args.load_str)
net_path = net_path + 'best' if args.load_epoch is None else net_path + str(args.load_epoch)
print(net_path)
net.load_state_dict(ch.load(net_path))

net.eval()

NUM_STEPS = args.num_steps
attack = pgd_l2 #if args.mode=='l2' else pgd_linf
print(attack)
EPS = args.eps
LR = args.pgd_lr if args.pgd_lr is not None else 2.*EPS/NUM_STEPS
print(LR)
NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

loss_fn = ch.nn.CrossEntropyLoss()


num_correct = 0
num_total = 0
for (images, labels) in testloader:
    images, labels = images.cuda(), labels.cuda()
    #images = ae.encode(images)#encode(images)
    attack_args = [net, images.clone(), labels, NUM_STEPS, LR, EPS, args.random_step]
    attack_images = attack(*attack_args)
    print("----")
    print(ch.norm((attack_images - images).view(images.shape[0],-1), dim=1))
    print("----")
    
    pred_probs = net(attack_images) # Shape: (BATCH_SIZE x 10)
    pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
    num_correct += (pred_classes == labels).float().sum()
    num_total += labels.shape[0]
    print(num_correct/num_total)

    xx = (pred_classes != labels)
    print(pred_classes[xx])
    print(labels[xx])
    print('')
print("###### EPOCH COMPLETE ######")
print("Adversarial Accuracy: %f" % (num_correct/num_total).cpu().item())
print("############################")
