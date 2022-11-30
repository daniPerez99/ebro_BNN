#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from random import randint
import numpy as np
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import pandas as pd
# Local imports
from config import *
from model import create_model, negative_loglikelihood
from plot import *
from callbacks import *


ITER = 'base'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# global parameters
process_data = False
normalize = 'None' #this value can be: 'minmax', 'mean' or None
optimizers = 'RMSprop' #this value can be: 'Adam' or 'SGD' or 'RMSprop'
seed = randint(0,100)
pred_date = '24H'
input = 18
layer1 = 8
layer2 = 4
epocas = 2375

# =============================================================================
# load and split data
# =============================================================================
if normalize == 'minmax':
    df = pd.read_csv('DATA/datos_procesados/'+ITER+'/datos_minmax_'+pred_date+'_NoRand.csv')
elif normalize == 'mean':
    df = pd.read_csv('DATA/datos_procesados/'+ITER+'/datos_mean_'+pred_date+'_NoRand.csv')
else:
    df = pd.read_csv('DATA/datos_procesados/'+ITER+'/datos_'+pred_date+'_NoRand.csv')

foldIter = 1
crossValRMSE = []
crossValMAE = []
crossValMaxError = []
crossValSTDD = []
crossValUncertainty = []
kf  = KFold(n_splits=5,shuffle=False)
for train,test in kf.split(df):
    X_train = df.iloc[train]
    X_test = df.iloc[test]
    y_test = X_test['pred']
    X_test = X_test.drop('pred',axis=1)
    y_train = X_train['pred']
    X_train = X_train.drop('pred',axis=1)
# =============================================================================
# create the model
# =============================================================================

    model = create_model(len(X_train), input, layer1, layer2)
    #model.summary()
    if optimizers == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif optimizers == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=0.01)
    elif optimizers == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(
            optimizer=optimizer,
            loss=negative_loglikelihood,
            metrics=[keras.metrics.RootMeanSquaredError()],

        )
# =============================================================================
# Train parameters
# =============================================================================
    print("#######################################################################")
    print("Training model fold nº {}, epoch {}, layer1 {}, layer2 {}, iter {}".format(foldIter,epocas,layer1,layer2,ITER))
    print("#######################################################################")
    print("Start training the model...")

    myCallback = PrintCallback()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="LOGS/{}/CV_{}_{}_{}".format(ITER,layer1,layer2,epocas))
    model.fit(X_train,y_train,epochs=epocas, verbose=0, use_multiprocessing=True,
        callbacks=[myCallback],
        validation_split=0.1,validation_freq=25)

    print("Training finished.")

    # =============================================================================
    # model evaluation
    # =============================================================================
    prediction_distributions = model(X_test.to_numpy())
    prediction_mean = prediction_distributions.mean().numpy().tolist()
    prediction_stdv = prediction_distributions.stddev().numpy()
    # The 95% CI is computed as mean (1.96 * stdv)
    upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
    lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
    prediction_stdv = prediction_stdv.tolist()
    size = len(y_test)  
    mean_error = np.zeros(size)
    y_test = y_test.to_numpy().tolist()

    uncertainty_counter = 0
    uncertainty_list = []
    max_errors = []
    for idx in range(size):
        mean_error[idx] = prediction_mean[idx][0] - y_test[idx]
        if y_test[idx] > upper[idx][0] or y_test[idx] < lower[idx][0]:
            uncertainty_counter += 1
            uncertainty_list.append(idx)
        if len(max_errors) < 10 or max_errors[0][0] < np.abs(mean_error[idx]):
            if len(max_errors) == 10:
                max_errors.remove(max_errors[0])
            max_errors.append([np.abs(mean_error[idx]),idx])
            max_errors.sort()

    RMSE = round(mean_squared_error(y_test,prediction_mean,squared=False), 2)
    crossValRMSE.append(RMSE)
    MAE = round(mean_absolute_error(y_test,prediction_mean),2)
    crossValMAE.append(MAE)
    MAXE = round(max_error(y_test,prediction_mean),2)
    crossValMaxError.append(MAXE)
    std = round(np.mean(prediction_stdv),2)
    crossValSTDD.append(std)
    crossValUncertainty.append(uncertainty_counter)
    # =============================================================================
    # plot the results
    # =============================================================================
    print(f"Root mean squared error: {RMSE}")
    print(f"Mean absolute error: {MAE}")
    print(f"Max error: {MAXE}")
    print(f"Number of uncertainy: {uncertainty_counter}")
    print(f"Mean of stddev: {std}")
    print(f"TOP 10 maximum errors:")
    for idx in range(10):
        print(
            f"Prediction mean: {round(prediction_mean[max_errors[idx][1]][0], 2)}, "
            f"stddev: {round(prediction_stdv[max_errors[idx][1]][0], 2)}, "
            f"95% CI: [{round(upper[max_errors[idx][1]][0], 2)} - {round(lower[max_errors[idx][1]][0], 2)}]"
            f" - Actual: {y_test[max_errors[idx][1]]}"
            f" - Mean error: {round(mean_error[max_errors[idx][1]], 2)}"
        )
    plot_data(uncertainty_list,y_test,prediction_mean,upper,lower,"./LOGS/plots/uncertainty.png")
    max_errors = [x[1] for x in max_errors]
    plot_data(max_errors,y_test,prediction_mean,upper,lower,"./LOGS/plots/maxError.png")
    
    foldIter += 1
print("#######################################################################")
print("Cross validation results:")
print("#######################################################################")
print(f"Root mean squared error: {round(np.mean(crossValRMSE),2)} +- {round(np.std(crossValRMSE),2)}")
print(f"Mean absolute error: {round(np.mean(crossValMAE),2)} +- {round(np.std(crossValMAE),2)}")
print(f"Max error: {round(np.mean(crossValMaxError),2)} +- {round(np.std(crossValMaxError),2)}")
print(f"Number of uncertainy: {round(np.mean(crossValUncertainty),2)} +- {round(np.std(crossValUncertainty),2)}")
print(f"Mean of stddev: {round(np.mean(crossValSTDD),2)} +- {round(np.std(crossValSTDD),2)}")
print("#######################################################################")
print("End of the program")