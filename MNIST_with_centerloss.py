import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets
from  torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*3*3, 2)
        self.ip2 = nn.Linear(2, 10)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*3*3)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        return ip1, F.log_softmax(ip2)


def train(train_loader, model, centers, criterion, optimizer, epoch):
    print "Training... Epoch = %d" % epoch
    ip1_loader = []
    idx_loader = []
    for i,(data, target) in enumerate(train_loader):
        if i < 10:
            data, target = Variable(data), Variable(target)
            ip1, pred = model(data)
            print i
            loss = criterion(pred, target) + CenterLoss(target, ip1, centers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print ip1, pred
            ip1_loader.append(ip1)
            idx_loader.append((target))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.numpy(),labels.data.numpy(),epoch)

def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.show()
    plt.pause(0.001)

def CenterLoss(y, feat, centers):
    centers_pred = centers.index_select(0, y.long())
    # print centers_pred.size()
    # print feat
    difference   = feat - centers_pred
    loss         = difference.pow(2).sum() / (2*  y.size()[0])
    return loss

def main():
    # Dataset
    trainset = datasets.MNIST('../../data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    # Model
    model = Net()

    # NLLLoss
    criterion = nn.NLLLoss()

    # init centers
    centers = Variable(torch.randn(10, 2).type(torch.FloatTensor), requires_grad=True)

    # 'centers' included
    optimizer = optim.Adam([{'params':model.parameters()},{'params': [centers]}])

    for epoch in range(5):
        train(train_loader, model, centers, criterion, optimizer, epoch+1)


if __name__ == '__main__':
    main()
