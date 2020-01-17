import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils_rnn import plot_seqs


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mda(y_true, y_pred, t=12):
    d = K.equal(K.sign(y_true[t: ] - y_true[:-t]),
                K.sign(y_pred[t: ] - y_pred[:-t]))
    return K.mean(K.cast(d, K.floatx()))

    

def build_SimpleRNN(input_shape, parameters):
    model = Sequential()
    model.add(SimpleRNN(units=parameters['RNN_size'], batch_input_shape=input_shape,
                        unroll=True, stateful=True))
    model.add(Dropout(parameters['dropout']))
    model.add(Dense(parameters['FC_size'], activation='relu'))
    model.add(Dense(1))
    optimizer = parameters["optim"](lr = parameters["lr"])
    model.compile(loss='mse', optimizer=optimizer, metrics=[rmse, mda])
    return model


def build_GRU(input_shape, parameters):
    model = Sequential()
    model.add(GRU(units=parameters['RNN_size'], batch_input_shape=input_shape,
                  unroll=True, stateful=True))
    model.add(Dropout(parameters['dropout']))
    model.add(Dense(parameters['FC_size'], activation='relu'))
    model.add(Dense(1))
    optimizer = parameters["optim"](lr = parameters["lr"])
    model.compile(loss='mse', optimizer=optimizer, metrics=[rmse, mda])
    return model


def build_LSTM(input_shape, parameters):
    model = Sequential()
    model.add(LSTM(units=parameters['RNN_size'], 
                   batch_input_shape=input_shape, 
                   unroll=True, stateful=True))
    model.add(Dropout(parameters['dropout']))
    model.add(Dense(parameters['FC_size'], activation='relu'))
    model.add(Dense(1))
    optimizer = parameters["optim"](lr = parameters["lr"])
    model.compile(loss='mse', optimizer=optimizer, metrics=[rmse, mda])
    return model


def training_callbacks(callback_lsit, params, filepath=None):
    callbacks = []
    if 'mcp' in callback_lsit:
        mcp = ModelCheckpoint(filepath=filepath, 
                              monitor='val_loss', verbose=1,
                              save_best_only=True, save_weights_only=False, 
                              mode='min', period=1)
        callbacks.append(mcp)
        
    if 'csv_logger' in callback_lsit:
        csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 
                                            'training_log_' + time.ctime().replace(" ","_") + '.log'), 
                               append=True)
        callbacks.append(csv_logger)

    if 'es' in callback_lsit:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                           patience = params["earlystop"]["patience"], 
                           min_delta = params["earlystop"]["min_delta"])
        callbacks.append(es)
        
    if 'reduce_lr' in callback_lsit:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                      factor = params["reduce_lr"]["factor"], 
                                      patience = params["reduce_lr"]["patience"],
                                      min_delta = params["reduce_lr"]["min_delta"],
                                      verbose=0, mode='auto' , 
                                      cooldown=0, min_lr=0)
        callbacks.append(reduce_lr)

    return callbacks

    
class ModelPredictions:
    def __init__(self,
                 model,
                 X,
                 y,
                 raw_y,
                 datetime,
                 window = 12,
                 batch_size = 32,
                 stnd_method = "standard"):
        self.__model = model
        self.__X = X
        self.__y = y
        self.__raw_y = raw_y
        self.__datetime = datetime
        self.__window = window
        self.__batch_size = batch_size
        self.__stnd_method = stnd_method

        if type(self.__X) is list:
            self.predictions = []
            self.predictions_org = []
            self.true = []
            self.true_org = []
            
            assert len(self.__X) == len(self.__y) == len(self.__datetime), \
            "Lists of input samples (X), target variables (y) and timestamp don't match in length!"
            
            start_index = 0
            for x, y in zip(self.__X, self.__y):
                assert x.shape[0] == y.shape[0], \
                "Input sample array (X) does not match target variable array (y)!"
                
                pred = pd.Series(self._predict(x, self.__model, self.__batch_size).reshape(-1),
                                 index = range(start_index, start_index + x.shape[0]))
                true = pd.Series(y.reshape(-1), index = range(start_index, start_index + y.shape[0]))
                pred_org = pd.Series(self._destnd(pred.values.reshape(-1,1),
                                                  self.__raw_y,
                                                  self.__stnd_method)
                                         .reshape(-1),
                                     index = pred.index)
                true_org = pd.Series(self._destnd(true.values.reshape(-1,1),
                                                  self.__raw_y,
                                                  self.__stnd_method)
                                         .reshape(-1),
                                     index = true.index)                                 
                self.predictions.append(pred)
                self.true.append(true)
                self.predictions_org.append(pred_org)
                self.true_org.append(true_org)
                start_index += x.shape[0] 

        else:
            assert X.shape[0] == y.shape[0], \
            "Input sample array (X) does not match target variable array (y)!"
            
            self.predictions = pd.Series(self._predict(self.__X, 
                                                       self.__model, 
                                                       self.__batch_size)
                                             .reshape(-1))
            self.true = pd.Series(self.__y.reshape(-1))
            self.predictions_org = pd.Series(self._destnd(self.predictions.values.reshape(-1,1),
                                                          self.__raw_y,
                                                          self.__stnd_method)
                                                 .reshape(-1))
            self.true_org = pd.Series(self._destnd(self.__y,
                                                   self.__raw_y,
                                                   self.__stnd_method)
                                          .reshape(-1))     
        
        self.loss = self.cal_mse(self.predictions_org, self.true_org)
        self.rmse = self.cal_rmse(self.predictions_org, self.true_org)
        self.mda = self.cal_mda(self.predictions_org, self.true_org, self.__window)
                
    def _predict(self, X, model, batch_size):
        pred = model.predict(X, batch_size)
        return pred
    
    
    def _destnd(self, seq_to_destnd, seq_fitting, method):
        if ('max' in method) or ('min' in method):
            scaler = MinMaxScaler()
            scaler.fit(seq_fitting)
        elif 'stand' in method:
            scaler = StandardScaler()
            scaler.fit(seq_fitting)
        seq_destnd = scaler.inverse_transform(seq_to_destnd)
        return seq_destnd
    
    
    def plot_predictions(self, labels=None, title=None, origin=True):
        if labels is not None:
            assert len(labels) - len(self.loss) == 1, \
            "The number of labels does NOT match the number of sequences to plot."

            for i in range(len(labels)):
                if i == 0:
                    labels[i] += " (MSE/ACC)"
                else:
                    labels[i] = labels[i] + " (" + \
                                str(round(self.loss[i-1], 2)) + "/" + \
                                str(round(self.mda[i-1], 2)) + ")"

        if type(self.__X) is list:
            dtime = pd.concat(self.__datetime, axis=0, ignore_index=True)
            if origin:
                true = pd.concat(self.true_org, axis=0, ignore_index=True)
                pred = pd.concat(self.predictions_org, axis=0, ignore_index=False)
                assert (true.index == pred.index).all(), \
                "Predictions does not match target variables!"
                assert (pred.index == dtime.index).all(), \
                "Predictions does not match timestamp!"
                plot_seqs([true]+self.predictions_org, dtime,
                          title=title, labels=labels)
            else:
                true = pd.concat(self.true, axis=0, ignore_index=True)
                pred = pd.concat(self.predictions, axis=0, ignore_index=False)
                assert (true.index == pred.index).all(), \
                "Predictions does not match target variables!"
                assert (pred.index == dtime.index).all(), \
                "Predictions does not match timestamp!"
                plot_seqs([self.true]+self.predictions, dtime,
                          title=title, labels=labels)
        else:
            if origin:
                assert (self.true_org.index == self.predictions_org.index).all(), \
                "Predictions does not match target variables!"
                assert (self.true_org.shape[0] == self.__datetime.shape[0]), \
                "Predictions does not match timestamp!"
                plot_seqs([self.true_org, self.predictions_org],
                          self.__datetime,
                          title=title, labels=labels)
            else:
                assert (self.true.index == self.predictions.index).all(), \
                "Predictions does not match target variables!"
                assert (self.true_org.index == self.__datetime.index), \
                "Predictions does not match timestamp!"
                plot_seqs([self.true, self.predictions], 
                          self.__datetime,
                          title=title, labels=labels)
            
    def cal_mse(self, pred, true):
        if type(pred) is list:
            mse = [(y - y_hat).pow(2).mean() for y, y_hat in zip(pred, true)]
        else:
            mse = [(pred - true).pow(2).mean()]
        return mse

    def cal_rmse(self, pred, true):
        if type(pred) is list:
            rmse = [np.sqrt((y - y_hat).pow(2).mean()) for y, y_hat in zip(pred, true)]
        else:
            rmse = [np.sqrt((pred - true).pow(2).mean())]
        return rmse
    
    def cal_mda(self, pred, true, t):
        if type(pred) is list:
            mda = [np.equal(np.sign(y[t::t].values - y[:-t:t].values), 
                            np.sign(y_hat[t::t].values - y_hat[:-t:t].values))
                     .mean() 
                   for y, y_hat in zip(pred, true)]
        else:
            mda = [np.equal(np.sign(pred[t::t].values - pred[:-t:t].values), 
                            np.sign(true[t::t].values - true[:-t:t].values))
                     .mean()]
        return mda        
        