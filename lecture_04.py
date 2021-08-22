import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def loss(w, x, y):
    y_predict = x * w
    return (y_predict - y) * (y_predict - y)

w = Variable(torch.Tensor([1.0]), requires_grad = True)
print("Inital w value :" + str(w.data))
alpha = 0.01

for step in range(100):
    sum = 0
    for x, y in zip(x_data, y_data):
        l = loss(w, x, y)
        sum += l
        l.backward()
        w.data = w.data - w.grad.data * alpha
    sum /= 3.0
    print("Step " + str(step) + " : w = " + str(w.data) + "  loss = " + str(sum))
    w.grad.data.zero_()
print(w.data * 3.0)
