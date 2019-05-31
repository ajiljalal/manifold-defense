from torchvision import datasets, transforms
import torch as ch
from torch.utils.data.dataset import Subset

BATCH_SIZE=50

image_transform = transforms.Compose([
#        transforms.RandomCrop(28, padding=4),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

fullset = datasets.MNIST(root=".data", train=True, transform=image_transform, download=True)
testset = datasets.MNIST(root=".data", train=False, transform=test_transform, download=True)

trainloaderfull = ch.utils.data.DataLoader(fullset, batch_size=BATCH_SIZE, shuffle=True)
testloader = ch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

trainset = Subset(fullset,range(55000))
valset = Subset(fullset, range(55000,60000))
trainloader = ch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validationloader = ch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

