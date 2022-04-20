#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Local imports
from config import *
from data import prepare_data
from plot import *
from model import create_model, negative_loglikelihood

# MAIN
# =============================================================================
# run the program with the CPU, comment the next line to use the GPU.
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

# =============================================================================
# process input data
# =============================================================================
df = prepare_data()
y = df['pred']
X = df.drop('pred',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=P_TRAIN)
X_train1 = X_train.drop(['arce','calatayud','ebro_y','laloteta','mansilla_y','romeral','tauste','tranquera_y','yesa_y'],axis=1)
X_test1 = X_test.drop(['arce','calatayud','ebro_y','laloteta','mansilla_y','romeral','tauste','tranquera_y','yesa_y'],axis=1)
# =============================================================================
# To load the weights previously trained
# =============================================================================

model1 = create_model(len(X_train1), 12, 5, 2)
model1.summary()
model1.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss=negative_loglikelihood,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

model1.load_weights('./MODEL/my_model_weights')

model2 = create_model(len(X_train), 21, 7, 3)
model2.summary()
model2.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss=negative_loglikelihood,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
model2.load_weights('./MODEL/my_model_weights2')

# =============================================================================
# model evaluation
# =============================================================================

prediction_distributions1 = model1(X_test1.to_numpy())
prediction_mean1 = prediction_distributions1.mean().numpy().tolist()
prediction_stdv1 = prediction_distributions1.stddev().numpy()
# The 95% CI is computed as mean (1.96 * stdv)
upper1 = (prediction_mean1 + (1.96 * prediction_stdv1)).tolist()
lower1 = (prediction_mean1 - (1.96 * prediction_stdv1)).tolist()
prediction_stdv1 = prediction_stdv1.tolist()
size = len(y_test)  
mean_error1 = np.zeros(size)


prediction_distributions2 = model2(X_test.to_numpy())
prediction_mean2 = prediction_distributions2.mean().numpy().tolist()
prediction_stdv2 = prediction_distributions2.stddev().numpy()
# The 95% CI is computed as mean (1.96 * stdv)
upper2 = (prediction_mean2 + (1.96 * prediction_stdv2)).tolist()
lower2 = (prediction_mean2 - (1.96 * prediction_stdv2)).tolist()
prediction_stdv2 = prediction_stdv2.tolist()
mean_error2 = np.zeros(size)
y_test = y_test.to_numpy().tolist()
# =============================================================================
# plot the results
# =============================================================================
for idx in range(5):
    mean_error1[idx] = y_test[idx] - prediction_mean1[idx][0]
    mean_error2[idx] = y_test[idx] - prediction_mean2[idx][0]
    print("--------------------------------------------------------------------------------")
    print(
        f"Prediction mean: {round(prediction_mean1[idx][0], 2)}, "
        f"stddev: {round(prediction_stdv1[idx][0], 2)}, "
        f"95% CI: [{round(upper1[idx][0], 2)} - {round(lower1[idx][0], 2)}]"
        f" - Actual: {y_test[idx]}"
        f" - Mean error: {round(mean_error1[idx], 2)}"
    )

    print(
        f"Prediction mean: {round(prediction_mean2[idx][0], 2)}, "
        f"stddev: {round(prediction_stdv2[idx][0], 2)}, "
        f"95% CI: [{round(upper2[idx][0], 2)} - {round(lower2[idx][0], 2)}]"
        f" - Actual: {y_test[idx]}"
        f" - Mean error: {round(mean_error2[idx], 2)}"
    )
    print("--------------------------------------------------------------------------------")


print(f"Root mean squared error 1: {round(mean_squared_error(y_test,prediction_mean1,squared=False), 2)}")
print(f"mean absolute error 1: {round(mean_absolute_error(y_test,prediction_mean1),2)}")
print(f"Root mean squared error 2: {round(mean_squared_error(y_test,prediction_mean2,squared=False), 2)}")
print(f"mean absolute error 2: {round(mean_absolute_error(y_test,prediction_mean2),2)}")
plot_data(y_test, prediction_mean1, upper1, lower1,'./LOGS/comparacion_1.png')
plot_data(y_test, prediction_mean2, upper2, lower2,'./LOGS/comparacion_2.png')