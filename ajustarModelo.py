# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
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
process_data = False
if process_data:
    df = prepare_data()
    df = shuffle(df)
    df.to_csv('DATA/datos_procesados/datos.csv',index=False)

df = pd.read_csv('DATA/datos_procesados/datos.csv')
y = df['pred']
X = df.drop('pred',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=P_TRAIN,shuffle=False)
y_test = y_test.to_numpy().tolist()
# =============================================================================
# create the model
# =============================================================================
for layer1 in range (1,NUM_FEATURES):
    for layer2 in range (1,layer1):
        model = create_model(len(X_train), NUM_FEATURES, layer1, layer2)
        model.summary()
        model.compile(
                optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
                loss=negative_loglikelihood,
                metrics=[keras.metrics.RootMeanSquaredError()],

            )
        # =============================================================================
        # Train parameters
        # =============================================================================
        print("Start training the model...")
        history = model.fit(X_train,y_train,epochs=NUM_EPOCHS, verbose=0,use_multiprocessing=True, 
                    callbacks=tf.keras.callbacks.EarlyStopping(patience=20,verbose=1),
                    validation_split=0.1,validation_freq=25)
        print("Training finished.")
        # Save model
        model.save("./MODEL/my_model")

        # la siguiente línea salva los pesos, que se pueden cargar más adelante. Pero el modelo no estoy seguro de que el modelo que se carga sea el mismo
        # los resultados varían ligeramente, pero podría ser por su naturaleza probabilística.  
        model.save_weights('./MODEL/my_model_weights')
        numEpoch = len(history.epoch)

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

        # =============================================================================
        # plot the results
        # =============================================================================

        print(f"Mean squared error: {round(mean_squared_error(y_test,prediction_mean), 2)}")
        print(f"Mean absolute error: {round(mean_absolute_error(y_test,prediction_mean),2)}")
        print(f"Max error: {round(max_error(y_test,prediction_mean),2)}")
        print(f"Number of uncertainy: {uncertainty_counter}")
        print(f"Number of epoch: {numEpoch}")
        print(f"TOP 10 maximum errors:")
        for idx in range(10):
            print(
                f"Prediction mean: {round(prediction_mean[max_errors[idx][1]][0], 2)}, "
                f"stddev: {round(prediction_stdv[max_errors[idx][1]][0], 2)}, "
                f"95% CI: [{round(upper[max_errors[idx][1]][0], 2)} - {round(lower[max_errors[idx][1]][0], 2)}]"
                f" - Actual: {y_test[max_errors[idx][1]]}"
                f" - Mean error: {round(mean_error[max_errors[idx][1]], 2)}"
            )
        plot_data(y_test, prediction_mean, upper, lower)

