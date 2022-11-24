# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(list,y_test,prediction_mean,upper,lower,fig_name='./LOGS/plots/plot.png'):
    y_test_plot = []
    prediction_mean_plot = []
    upper_plot = []
    lower_plot = []
    for idx in list:
        y_test_plot.append(y_test[idx])
        prediction_mean_plot.append(prediction_mean[idx][0])
        upper_plot.append(upper[idx][0])
        lower_plot.append(lower[idx][0])

    num_print = len(y_test_plot)

    #plot the results
    plt.stem(range(num_print),prediction_mean_plot[0:num_print], markerfmt='bo', basefmt='k-', label='prediction')
    plt.stem(range(num_print),y_test_plot[0:num_print], markerfmt='ro', basefmt='k-', label='real')
    #plt.plot(range(num_print),prediction_mean[0:num_print], label='prediction')
    #plt.plot(range(num_print),y_test[0:num_print], label='real')
    plt.fill_between(range(num_print),lower_plot[0:num_print],upper_plot[0:num_print], alpha=0.5,color='g')
    plt.legend()
    plt.savefig(fig_name)
    plt.close()
def histograma():
    data = pd.read_csv("./DATA/datos_procesados/base/datos_24H.csv")
    plt.hist(x=data['pred'],bins=range(7),rwidth=0.85)
    plt.title("Histograma de nivel del ebro")
    plt.xlabel("nivel (m)")
    plt.ylabel("casos")
    plt.savefig("./LOGS/plots/histograma.png")

#histograma()