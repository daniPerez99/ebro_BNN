import tensorflow as tf
import time

class PrintCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, save_epoch=500, print_epoch=10, earlyStop = False, monitor='val_loss', 
                    patience=20, checkpoint_path= "LOGS/train/", restore_best_model = False,
                    validation_freq = 1):
        self.save_epoch = save_epoch
        self.print_epoch = print_epoch   
        self.early_stop = earlyStop
        self.restore_best_model = restore_best_model
        self.checkpoint_path = checkpoint_path
        self.monitor = monitor
        self.patience = patience * validation_freq


    def print_loss_acc(self, logs, last=False, time=None):
        loss = sum(self.losses[-self.print_epoch:])/self.print_epoch
        rmse = sum(self.rmse[-self.print_epoch:])/self.print_epoch
        if last:
            print("\n--- TRAIN END AT EPOCH {} ---".format(self.epoch))
            print("TRAINING TIME: {} seconds".format(time))
            print("Best epoch: {}".format(self.best_epoch))
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
        self.best_epoch = 0
        self.best_loss = None
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.rmse.append(logs.get('root_mean_squared_error'))
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if self.epoch % self.print_epoch == 0:
            self.print_loss_acc(logs)
        if self.early_stop:
            #save the best model
            new_loss = logs.get(self.monitor)
            if new_loss is not None:
                #new_loss = abs(new_loss)
                if self.best_loss is None or self.best_loss > new_loss:
                    self.best_loss = new_loss
                    if self.restore_best_model:
                        self.model.save(self.checkpoint_path)
                    self.best_epoch = self.epoch
                #stop training if the model is not improving
                if self.best_epoch == self.epoch - self.patience:
                    self.model.stop_training = True

    def on_train_end(self, logs={}):
        total_time = time.time() - self.start_time
        self.print_loss_acc(logs, last=True, time=total_time)
