import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import trange


class DiabetesDataset(Dataset):

    def __init__(self):
        file = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = file.shape[0]
        x_raw = file[:, 0:-1]
        y_raw = file[:, [-1]]
        self.x_data = torch.from_numpy(x_raw)
        self.y_data = torch.from_numpy(y_raw)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        out1 = F.sigmoid(self.linear1(x))
        out2 = F.sigmoid(self.linear2(out1))
        y_pred = F.sigmoid(self.linear3(out2))
        return y_pred


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.9)

for epoch in trange(30000):
    for data in train_loader:
        x_data, y_data = data
        x_data, y_data = Variable(x_data), Variable(y_data)
        y_pred = model.forward(x_data)
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

file = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_test = Variable(torch.tensor(file[:, 0:-1]))
y_test = Variable(torch.tensor(file[:, [-1]]))
y_predict = model(x_test)
accuracy = 0
n = len(y_predict)
for i in range(0, n):
    if (int(y_predict.data[i][0] > 0.5) == y_test.data[i][0]):
        accuracy += 1
print(accuracy / n)
