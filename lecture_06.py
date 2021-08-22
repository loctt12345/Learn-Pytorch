import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

x_raw = [[x] for x in np.arange(-10.0, 10.0, 0.5)]
x_raw.append([30])
y_raw = [[1 if x_raw[x][0] > 0 else 0] for x in range(40)]
y_raw.append([0])
x_data = Variable(torch.Tensor(x_raw))
y_data = Variable(torch.Tensor(y_raw))

class Model(torch.nn.Module) :
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for i in range(41) :
    shape = ''
    if (y_raw[i][0] == 0) :
        shape = 'bo'
    else :
        shape = 'r+'
    plt.plot(x_raw[i][0], y_raw[i][0], shape)
plt.plot(list(np.arange(-10.0, 10.0, 0.5)) + [30], model(x_data).data[:,0].tolist())
plt.xlabel("x")
plt.ylabel("y predict")
plt.show()