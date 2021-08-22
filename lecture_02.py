import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_list = np.arange(0.1, 0.4, 0.1)

def loss(w, x, y) :
    y_predict = x * w
    return (y_predict - y) * (y_predict - y)
loss_list = []
for w in w_list :
    loss_sum = 0
    for x, y in zip(x_data, y_data) :
        loss_sum += loss(w, x, y)
    loss_list.append(loss_sum / 3.0)
plt.plot(w_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('W')
plt.show()