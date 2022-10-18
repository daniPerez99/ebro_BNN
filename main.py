from calendar import EPOCH
import os
from random import randint
import time
import numpy as np
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
import pandas as pd
# Local imports
from config import *
from data import prepare_data, normalize_data_minmax, normalize_data_mean
from model import create_model, negative_loglikelihood
from plot import *


# CALLBACK FUNCTION
# =============================================================================

class PrintCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, save_epoch=500, print_epoch=10):
        self.save_epoch = save_epoch
        self.print_epoch = print_epoch
    
    def print_loss_acc(self, logs, last=False, time=None):
        loss = sum(self.losses[-self.print_epoch:])/self.print_epoch
        rmse = sum(self.rmse[-self.print_epoch:])/self.print_epoch
        if last:
            print("\n--- TRAIN END AT EPOCH {} ---".format(self.epoch))
            print("TRAINING TIME: {} seconds".format(time))
            print("Epoch loss ({}): {} - RMSE: {}".format(self.epoch, loss,rmse),end='\n')
        else:
            if self.epoch % self.save_epoch == 0:
                loss = sum(self.losses[-self.save_epoch:])/self.save_epoch
                rmse = sum(self.rmse[-self.save_epoch:])/self.save_epoch
                print("Epoch loss ({}): {} - RMSE: {}".format(self.epoch, loss,rmse),end='\n')
            else:
                print("Epoch loss ({}): {} - RMSE: {}".format(self.epoch, loss,rmse),end='\r')

    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch = 0
        self.start_time = time.time()
        self.rmse = []
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.rmse.append(logs.get('root_mean_squared_error'))
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if self.epoch % self.print_epoch == 0:
            self.print_loss_acc(logs)
    
    def on_train_end(self, logs={}):
        total_time = time.time() - self.start_time
        self.print_loss_acc(logs, last=True, time=total_time)

# MAIN
# =============================================================================
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # global parameters
    process_data = False
    normalize = 'None' #this value can be: 'minmax', 'mean' or None
    train_model = True
    save_model = False
    load_model = True
    train_eval = 'val_loss' #this value can be: 'val_root_mean_squared_error' or 'val_loss'
    optimizers = 'SGD'
    for pred_date in ['24H', '48H', '72H']:
        checkpoint_filepath = 'MODEL/checkpoint/'+normalize+'_'+pred_date+'_'+str(LAYER1_NEURONS)+'_'+str(LAYER2_NEURONS)+'_'+str(NUM_EPOCHS)+'_'+optimizers+'_'+train_eval
        # =============================================================================
        # process input data
        # =============================================================================
        if process_data:
            seed = randint(0,100)
            df = prepare_data()
            #the shuffle is done here to have the same distribution of data in all cases.
            df = shuffle(df, random_state=seed)
            
            df.to_csv('DATA/datos_procesados/datos_'+pred_date+'.csv',index=False)
            
            df_aux = normalize_data_mean(df)
            df_aux.to_csv('DATA/datos_procesados/datos_mean_'+pred_date+'.csv',index=False)

            df_aux = normalize_data_minmax(df)
            df.to_csv('DATA/datos_procesados/datos_minmax_'+pred_date+'.csv',index=False)

        # =============================================================================
        # load and split data
        # =============================================================================
        if normalize == 'minmax':
            df = pd.read_csv('DATA/datos_procesados/datos_minmax_'+pred_date+'.csv')
        elif normalize == 'mean':
            df = pd.read_csv('DATA/datos_procesados/datos_mean_'+pred_date+'.csv')
        else:
            df = pd.read_csv('DATA/datos_procesados/datos_'+pred_date+'.csv')

        y = df['pred']
        X = df.drop('pred',axis=1)
        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=P_TRAIN,shuffle=False)
        # =============================================================================
        # create the model
        # =============================================================================

        model = create_model(len(X_train), NUM_FEATURES, LAYER1_NEURONS, LAYER2_NEURONS)
        #model.summary()
        if optimizers == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        elif optimizers == 'SGD':
            optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        elif optimizers == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
        model.compile(
                optimizer=optimizer,
                loss=negative_loglikelihood,
                metrics=[keras.metrics.RootMeanSquaredError()],

            )
        # =============================================================================
        # Train parameters
        # =============================================================================
        if train_model:
            print("#######################################################################")
            print("Training model with {} neurons in layer 1 and {} neurons in layer 2".format(LAYER1_NEURONS, LAYER2_NEURONS))
            print("#######################################################################")
            print("Start training the model...")
            checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor=train_eval, verbose=0, save_best_only=True)
            tensorboard = keras.callbacks.TensorBoard(log_dir='LOGS/train/'+normalize+'_'+pred_date+'_'+str(LAYER1_NEURONS)+'_'+str(LAYER2_NEURONS)+'_'+str(NUM_EPOCHS)+'_'+optimizers+'_'+train_eval),
            early_stop = keras.callbacks.EarlyStopping(monitor=train_eval, patience=20)
            model.fit(X_train,y_train,epochs=NUM_EPOCHS, verbose=0, use_multiprocessing=True,
                callbacks=[checkpoint,tensorboard,PrintCallback(),early_stop,],
                validation_split=0.1,validation_freq=25)
            print("Training finished.")
            # Save model
            if save_model:
                #model.save("./MODEL/my_model_"+normalize+'_'+pred_date)
                # la siguiente línea salva los pesos, que se pueden cargar más adelante. Pero el modelo no estoy seguro de que el modelo que se carga sea el mismo
                # los resultados varían ligeramente, pero podría ser por su naturaleza probabilística.  
                model.save_weights('./MODEL/my_model_weights_'+normalize+'_'+pred_date)

        # =============================================================================
        # To load the weights previously trained
        # =============================================================================

        if load_model:
            
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

        # =============================================================================
        # plot the results
        # =============================================================================
        print(f"Mean squared error: {round(mean_squared_error(y_test,prediction_mean), 2)}")
        print(f"Mean absolute error: {round(mean_absolute_error(y_test,prediction_mean),2)}")
        print(f"Max error: {round(max_error(y_test,prediction_mean),2)}")
        print(f"Number of uncertainy: {uncertainty_counter}")
        print(f"Mean of variance: {round(np.mean(prediction_stdv),2)}")
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


if __name__ == "__main__":
    main()