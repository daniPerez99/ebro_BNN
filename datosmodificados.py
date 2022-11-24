from operator import truediv
import sys
import os
from random import randint
import numpy as np
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split,KFold
import pandas as pd
# Local imports
from config import *
from data import prepare_data, normalize_data_minmax, normalize_data_mean, split_flood
from model import create_model, negative_loglikelihood
from plot import *
from callbacks import *
# MAIN
# =============================================================================

ITER = 'base'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# global parameters
pred_date = '24H'
input = 18
layer1 = 8
layer2 = 4
epocas = 2375
# =============================================================================
# process input data
    # =============================================================================

df = pd.read_csv('DATA/datos_procesados/'+ITER+'/datos_'+pred_date+'_NoRand.csv')
X_flood,X_no_flood  = split_flood(df)
#X_flood.to_csv('DATA/datos_procesados/'+ITER+'/datos_flood_'+pred_date+'.csv',index=False)
#X_no_flood.to_csv('DATA/datos_procesados/'+ITER+'/datos_no_flood_'+pred_date+'.csv',index=False)
X_overFit, X_test, y_overFit, y_test = train_test_split(X_flood, X_flood['pred'], test_size=0.5,random_state=1)
X_train = X_no_flood
for i in range(10):
    print(i)
    X_train = pd.concat([X_train,X_overFit])
X_train = shuffle(X_train)
y_train = X_train['pred']
X_train = X_train.drop('pred',axis=1)
X_test = X_flood
y_test = X_test['pred']
X_test = X_test.drop('pred',axis=1)

# =============================================================================
# create the model
# =============================================================================

model = create_model(len(X_train), input, layer1, layer2)

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
print("Training model epoch {}, layer1 {}, layer2 {}, iter {}".format(epocas,layer1,layer2,ITER))
print("#######################################################################")
print("Start training the model...")

myCallback = PrintCallback()
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
MAE = round(mean_absolute_error(y_test,prediction_mean),2)
MAXE = round(max_error(y_test,prediction_mean),2)
std = round(np.mean(prediction_stdv),2)
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
