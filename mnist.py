from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from dead_op import DeadConv2d
import torchsnooper # NOTE: @torchsnooper.snoop() 很有用，放在forward上面可以直接查看各个步骤的tensor大小，batchsize也能看的见， 放在自定义的函数上面可以看见各个步骤的大小
from torch import autograd # NOTE:with autograd.detect_anomaly():  RuntimeError: Function 'LogSoftmaxBackward' returned nan values in its 0th output. 这行可以直接报告哪一个函数出现了Nan
import numpy as np
np.random.seed(0)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

writer = SummaryWriter(comment='dead_op_trim_detail')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = DeadConv2d(in_channels = 1, out_channels = 20, kernel_size=[5,5])  
        self.conv2 = DeadConv2d(in_channels = 20, out_channels = 50, kernel_size=[5,5])
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)    # NOTE:输入通道， 输出通道， 卷积核宽度， stride
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        # self.one = nn.Parameter(torch.ones(500).cuda())

    # @torchsnooper.snoop()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def trim(args, model, device, train_loader, optimizer, epoch):
    with autograd.detect_anomaly():
        model.train()

        for param in model.parameters():
            param.requires_grad = False

        model.conv1.dead_lock.requires_grad = True
        model.conv2.dead_lock.requires_grad = True


        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    writer.add_histogram(name, param, epoch)

            
            writer.add_histogram("conv1.dead_lock", model.conv1.dead_lock, epoch)    
            writer.add_histogram("conv2.dead_lock", model.conv2.dead_lock, epoch)    

            if batch_idx % args.log_interval == 0:
                print('Trim Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        for param in model.parameters():
            param.requires_grad = True

        model.conv1.dead_lock.requires_grad = False
        model.conv2.dead_lock.requires_grad = False

    return loss.item()


def train(args, model, device, train_loader, optimizer, epoch):
    with autograd.detect_anomaly():
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    writer.add_histogram(name, param, epoch)

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_loss, test_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(args, model, device, test_loader)
        trim_loss = trim(args, model, device, train_loader, optimizer, epoch)
        test_trim_loss, test_trim_acc = test(args, model, device, test_loader)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_trim_loss', test_trim_loss, epoch)
        writer.add_scalar('test_trim_acc', test_trim_acc, epoch)
        writer.add_scalar('trim_loss', trim_loss, epoch)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
    writer.close()

if __name__ == '__main__':
    main()