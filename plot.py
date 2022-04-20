#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def plot_data(y_test, prediction_mean, upper, lower,fig_name='./LOGS/plot.png'):
    plt.stem(range(100),prediction_mean[0:100], markerfmt='bo', basefmt='k-')
    plt.stem(range(100),y_test[0:100], markerfmt='ro', basefmt='k-')
    y1 = []
    y2 = []
    for i in range(100):
        y1.append(lower[i][0])
        y2.append(upper[i][0])
    plt.fill_between(range(100),y1, y2, alpha=0.5,color='g')
    plt.savefig(fig_name)