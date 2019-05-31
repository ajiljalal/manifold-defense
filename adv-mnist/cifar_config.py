from torchvision import transforms, datasets
import torch as ch
from torch.utils.data.dataset import Subset

# We have to load up both the training set, and the test set separately
image_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

no_normalize_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

no_normalize_test_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

BATCH_SIZE = 50

fullset = datasets.CIFAR10(root=".data", train=True, transform=image_transform, download=True)
trainset = Subset(fullset, range(40000))
valset = Subset(fullset, range(40000,50000))
testset = datasets.CIFAR10(root=".data", train=False, transform=test_transform, download=True)
fullloader = ch.utils.data.DataLoader(fullset, batch_size=32, shuffle=True)
trainloader = ch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
validationloader = ch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)
testloader = ch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

no_norm_fullset = datasets.CIFAR10(root=".data", train=True, transform=no_normalize_transform, download=True)
no_norm_testset = datasets.CIFAR10(root=".data", train=False, transform=no_normalize_test_transform, download=True)
no_norm_trainset = Subset(no_norm_fullset, range(40000))
no_norm_valset = Subset(no_norm_fullset, range(40000,50000))

no_norm_fullloader = ch.utils.data.DataLoader(no_norm_fullset, batch_size=BATCH_SIZE, shuffle=True)
no_norm_trainloader = ch.utils.data.DataLoader(no_norm_trainset, batch_size=BATCH_SIZE, shuffle=True)
no_norm_validationloader = ch.utils.data.DataLoader(no_norm_valset, batch_size=BATCH_SIZE, shuffle=True)
no_norm_testloader = ch.utils.data.DataLoader(no_norm_testset, batch_size=BATCH_SIZE, shuffle=False)
