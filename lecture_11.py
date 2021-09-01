import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import trange


batch_size = 64
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Inception(torch.nn.Module):

    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.branch5x5_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3_1 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_2 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.branch_pool = torch.nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(branch3x3)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)
        self.incept1 = Inception(in_channels=10)
        self.incept2 = Inception(in_channels=20)
        self.mp = torch.nn.MaxPool2d(2)
        self.linear = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)
        x = self.linear(x)
        return F.log_softmax(x)

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(10):
    print(epoch, '%')
    for x_data, y_data in train_loader:
        optimizer.zero_grad()
        x_data, y_data = Variable(x_data), Variable(y_data)
        y_pred = model.forward(x_data)
        loss = criterion(y_pred, y_data)
        loss.backward()
        optimizer.step()

accuracy = 0
for x_test, y_test in test_loader:
        x_test, y_test = Variable(x_test), Variable(y_test)
        y_pred_val = model.forward((x_test))
        y_pred = y_pred_val.data.max(1, keepdim=True)[1]
        accuracy += y_pred.eq(y_test.data.view_as(y_pred)).cpu().sum()
print((accuracy / len(test_loader.dataset)) * 100, '%')