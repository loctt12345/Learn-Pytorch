import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def loss(w, x, y):
    y_predict = x * w
    return (y_predict - y) * (y_predict - y)

def calculate_der(w, x, y):
    return 2 * x * (x * w - y)

w_list = []
loss_list = []
w = 1.0
alpha = 0.01

for step in range(100):
    sum = 0
    for x, y in zip(x_data, y_data):
        sum += loss(w, x, y)
        der = calculate_der(w, x, y)
        w = w - der * alpha
    sum /= 3.0
    print("Step " + str(step) + " : w = " + str(w) + "  loss = " + str(sum))
    w_list.append(w)
    loss_list.append(loss)

