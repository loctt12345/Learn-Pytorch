import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

file = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_raw = file[0:650, 0:-1]
y_raw = file[0:650, [-1]]
x_data = Variable(torch.Tensor(x_raw))
y_data = Variable(torch.Tensor(y_raw))

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 7)
        self.linear2 = torch.nn.Linear(7, 5)
        self.linear3 = torch.nn.Linear(5, 4)
        self.linear4 = torch.nn.Linear(4, 6)
        self.linear5 = torch.nn.Linear(6, 5)
        self.linear6 = torch.nn.Linear(5, 3)
        self.linear7 = torch.nn.Linear(3, 4)
        self.linear8 = torch.nn.Linear(4, 2)
        self.linear9 = torch.nn.Linear(2, 3)
        self.linear10 = torch.nn.Linear(3, 1)

    def forward(self, x):
        out1 = F.sigmoid(self.linear1(x))
        out2 = F.sigmoid(self.linear2(out1))
        out3 = F.sigmoid(self.linear3(out2))
        out4 = F.sigmoid(self.linear4(out3))
        out5 = F.sigmoid(self.linear5(out4))
        out6 = F.sigmoid(self.linear6(out5))
        out7 = F.sigmoid(self.linear7(out6))
        out8 = F.sigmoid(self.linear8(out7))
        out9 = F.sigmoid(self.linear9(out8))
        y_pred = F.sigmoid(self.linear10(out9))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.9)

for epoch in trange(50000):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    #print(epoch, "Loss :" + str(loss.data))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_test = Variable(torch.tensor(file[650:-1, 0:-1]))
y_test = Variable(torch.tensor(file[650:-1, [-1]]))
y_predict = model(x_test)
accuracy = 0
n = len(y_predict)
for i in range(0, n):
    if (int(y_predict.data[i][0] > 0.5) == y_test.data[i][0]):
        accuracy += 1
print(accuracy / n)
