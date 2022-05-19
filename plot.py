#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def plot_data(y_test, prediction_mean, upper, lower,fig_name='./LOGS/plot.png'):
    plt.stem(range(10),prediction_mean[0:10], markerfmt='bo', basefmt='k-', label='prediction')
    plt.stem(range(10),y_test[0:10], markerfmt='ro', basefmt='k-', label='real')
    plt.legend()
    y1 = []
    y2 = []
    for i in range(10):
        y1.append(lower[i][0])
        y2.append(upper[i][0])
    plt.fill_between(range(10),y1, y2, alpha=0.5,color='g')
    plt.savefig(fig_name)
    plt.close()