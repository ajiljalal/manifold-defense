import torch
import torch.optim as optim
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
from models import Classifier

cla = Classifier().cuda()
opt = optim.Adam(cla.parameters(), lr=1e-3)

## Step 1: train the classifier
trainloader = data.DataLoader(
	datasets.MNIST('../data', download=True, train=True, transform=transforms.Compose([
		   transforms.ToTensor(),
	   ])), batch_size=100, shuffle=True)

testloader = data.DataLoader(
	datasets.MNIST('../data', download=True, train=False, transform=transforms.Compose([
		   transforms.ToTensor(),
	   ])), batch_size=100, shuffle=True)

def accuracy(): 
    total_correct = 0.0
    total = 0.0
    with torch.no_grad():
        for images, targets in testloader:
            out = cla(images.cuda())
            preds = out.argmax(1)
            total_correct += (preds.cpu()==targets).float().sum()
            total += preds.shape[0]
    return total_correct/total

loss_fn = nn.CrossEntropyLoss()
for _ in range(5):
    cla.train(True)
    for images, targets in trainloader:
        opt.zero_grad()
        out = cla(images.cuda())
        loss = loss_fn(out, targets.cuda())
        loss.backward()
        opt.step()
    print("Loss: %f" % (loss,))
    cla.eval()
    print(accuracy())

torch.save(cla, "ckpts/classifier")
