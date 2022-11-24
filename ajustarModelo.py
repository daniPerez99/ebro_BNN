# -*- coding: utf-8 -*-
import os
import sys
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
from callbacks import *

# MAIN
# =============================================================================
# run the program with the CPU, comment the next line to use the GPU.
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

# =============================================================================
# process input data
# =============================================================================
df = pd.read_csv('DATA/datos_procesados/'+ITER+'/datos_24H.csv')
y = df['pred']
X = df.drop('pred',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=P_TRAIN,shuffle=False)
y_test = y_test.to_numpy().tolist()
# =============================================================================
# create the model
# =============================================================================
optimizer = 'RMSprop'
for layer1 in range(1,10):
    for layer2 in range(1,layer1):
        tf.keras.backend.clear_session()
        model = create_model(len(X_train), NUM_FEATURES, layer1, layer2)
        if(optimizer == 'Adam'):
            opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        elif(optimizer == 'SGD'):
            opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        else:
            opt = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
        model.compile(
                optimizer=opt,
                loss=negative_loglikelihood,
                metrics=[keras.metrics.RootMeanSquaredError(),keras.metrics.KLDivergence()],

            )
        # =============================================================================
        # Train parameters
        # =============================================================================
        print("########################################################################################")
        print("Training model with {} optimizer, {} neurons in layer 1 and {} neurons in layer 2".format(optimizer,layer1,layer2))
        print("########################################################################################")
        print("Start training the model...")
        checkpoint_filepath = './MODEL/'+ITER+'/'+str(layer1)+'_'+str(layer2)+'_'+optimizer+'_'+str(LEARNING_RATE)
        tensorboard = keras.callbacks.TensorBoard(log_dir='LOGS/'+ITER+'/'+str(layer1)+
                                                    '_'+str(layer2)+'_'+optimizer+'_'+str(LEARNING_RATE))
        myCallback = PrintCallback(earlyStop=True,checkpoint_path=checkpoint_filepath,monitor='val_loss',
                                    restore_best_model=False,validation_freq=25,print_epoch=500)

        history = model.fit(X_train,y_train,epochs=NUM_EPOCHS, verbose=0,use_multiprocessing=True, 
                            callbacks=[myCallback,tensorboard],
                            validation_split=0.1,validation_freq=25,)
        print("Training finished.")
        model.save(checkpoint_filepath)
        # load model
        model.load_weights(checkpoint_filepath+'/variables/variables').expect_partial()
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

        print(f"Root Mean squared error: {round(mean_squared_error(y_test,prediction_mean,squared=False), 2)}")
        print(f"Mean absolute error: {round(mean_absolute_error(y_test,prediction_mean),2)}")
        print(f"Max error: {round(max_error(y_test,prediction_mean),2)}")
        print(f"Number of uncertainy: {uncertainty_counter}")
        print(f"Mean of stdv: {round(np.mean(prediction_stdv),2)}")
        print(f"TOP 10 maximum errors:")
        for idx in range(10):
            print(
                f"Prediction mean: {round(prediction_mean[max_errors[idx][1]][0], 2)}, "
                f"stddev: {round(prediction_stdv[max_errors[idx][1]][0], 2)}, "
                f"95% CI: [{round(upper[max_errors[idx][1]][0], 2)} - {round(lower[max_errors[idx][1]][0], 2)}]"
                f" - Actual: {y_test[max_errors[idx][1]]}"
                f" - Mean error: {round(mean_error[max_errors[idx][1]], 2)}"
            )

