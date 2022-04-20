#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Local imports
from config import *
from data import prepare_data
from model import create_model, negative_loglikelihood
from plot import *

# CALLBACK FUNCTION
# =============================================================================

class PrintCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, print_epoch=1000, losses_avg_no=100):
        self.print_epoch = print_epoch
        self.losses_avg_no = losses_avg_no
    
    def print_loss_acc(self, logs, last=False, time=None):
        loss = sum(self.losses[-self.losses_avg_no:])/self.losses_avg_no
        if last:
            print("\n--- TRAIN END AT EPOCH {} ---".format(self.epoch))
            print("TRAINING TIME: {} seconds".format(time))
        print("Epoch loss ({}): {}".format(self.epoch, loss))
#        print("Accuracy: {}".format(logs.get('val_accuracy')))
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch = 0
        self.start_time = time.time()
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if self.epoch % self.print_epoch == 0:
            self.print_loss_acc(logs)
    
    def on_train_end(self, logs={}):
        total_time = time.time() - self.start_time
        self.print_loss_acc(logs, last=True, time=total_time)

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
#X_train = X_train.drop(['arce','calatayud','ebro_y','laloteta','mansilla_y','romeral','tauste','tranquera_y','yesa_y'],axis=1)
#X_test = X_test.drop(['arce','calatayud','ebro_y','laloteta','mansilla_y','romeral','tauste','tranquera_y','yesa_y'],axis=1)
# =============================================================================
# create the model
# =============================================================================

model = create_model(len(X_train), NUM_FEATURES, LAYER1_NEURONS, LAYER2_NEURONS)
model.summary()
model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss=negative_loglikelihood,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
# =============================================================================
# Train parameters
# =============================================================================
train_model = False
if train_model:
    print("Start training the model...")
    model.fit(X_train,y_train,epochs=NUM_EPOCHS, verbose=0, use_multiprocessing=True, 
                callbacks=[PrintCallback(PRINT_EPOCH, LOSSES_AVG_NO)],
                validation_split=0.1,validation_freq=25)
    print("Training finished.")
    # Save model
    #model.save("./MODEL/my_model")
    model.save("./MODEL/my_model2")
    # la siguiente línea salva los pesos, que se pueden cargar más adelante. Pero el modelo no estoy seguro de que el modelo que se carga sea el mismo
    # los resultados varían ligeramente, pero podría ser por su naturaleza probabilística.  
    #model.save_weights('./MODEL/my_model_weights')
    model.save_weights('./MODEL/my_model_weights2')

# =============================================================================
# To load the weights previously trained
# =============================================================================

load_model = True
if load_model:
    #model.load_weights('./MODEL/my_model_weights')
    model.load_weights('./MODEL/my_model_weights2')

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
# =============================================================================
# plot the results
# =============================================================================

for idx in range(5):
    mean_error[idx] = y_test[idx] - prediction_mean[idx][0]
    print(
        f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
        f"stddev: {round(prediction_stdv[idx][0], 2)}, "
        f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
        f" - Actual: {y_test[idx]}"
        f" - Mean error: {round(mean_error[idx], 2)}"
    )

print(f"Root mean squared error: {round(mean_squared_error(y_test,prediction_mean,squared=False), 2)}")
print(f"mean absolute error: {round(mean_absolute_error(y_test,prediction_mean),2)}")
plot_data(y_test, prediction_mean, upper, lower)