import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from auto_LiRPA.utils import Flatten
from VRCP.utils import ConvNet

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from train_utils import progress_bar

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to be used: CIFAR100, CIFAR10, TINYNET')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

dataset = args.dataset

dim = 32
if dataset == "TINYNET":
    dim = 64

transform_train = transforms.Compose([
    transforms.RandomCrop(dim, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        transform_train(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_valid(example_batch):
    """Apply valid_transforms across a batch."""
    example_batch["pixel_values"] = [
        transform_test(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

if dataset == "TINYNET":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load train set
    trainset = torchvision.datasets.ImageFolder(root='tiny-imagenet-200/train', transform=transform)
    # Load test set
    testset = torchvision.datasets.ImageFolder(root='tiny-imagenet-200/val', transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

elif dataset == "CIFAR10":
    trainset = torchvision.datasets.CIFAR10(
        root='./Datasets', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./Datasets', train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
else:
    trainset = torchvision.datasets.CIFAR100(
            root='./Datasets', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./Datasets', train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')

if dataset == "CIFAR10":
    ckptPath = "Checkpoints/cifar10CNN.pth"
    net = nn.Sequential(
                nn.Conv2d(3, 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(512,1024),
                nn.ReLU(),
                nn.Linear(1024, 10)
    )

elif dataset == "CIFAR100":
    ckptPath = "Checkpoints/cifar100CNN.pth"
    net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Dropout(0.4),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Dropout(0.4),
                Flatten(),
                nn.Linear(1024,256),
                nn.ReLU(),
                nn.Linear(256, 1024)
            )
else:
    ckptPath = "Checkpoints/tinynetCNN.pth"
    net = ConvNet()

net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./'+ckptPath)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('Checkpoints'):
            os.mkdir('Checkpoints')
        torch.save(state, './'+ckptPath)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
