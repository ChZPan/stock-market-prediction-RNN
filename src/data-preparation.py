import os
import numpy as np
import pandas as pd
import random
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# random.seed(time.time())

class StockIndexDataset:
    def __init__(self,
                 dataset_files,
                 index_name="SH300IF",
                 batch_size=32,
                 time_steps=12,
                 forecast_steps=0,
                 val_ratio=0.2,
                 test_ratio=0.2,
                 standardization='standard',
                 close_price_only=True):
        self.dataset_files = dataset_files
        self.index_name = index_name
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.forecast_steps = forecast_steps
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.standardization = standardization

        # Read raw dataset file
        if type(self.dataset_files) is list:
            dfs = []
            for file in self.dataset_files:
                df = pd.read_table(file, delim_whitespace=True,
                                   names=('date', 'time', 'open', 'close'))
                dfs.append(df)
        
            self.raw_df = pd.concat(dfs, axis=0, ignore_index=True)
        else:
            self.raw_df = pd.read_table(self.dataset_files,
                                        delim_whitespace=True,
                                        names=('date', 'time', 'open', 'close'))
        
        datetime = pd.to_datetime(self.raw_df['date'] + ' ' + self.raw_df['time'])
        
        if close_price_only:
            self.raw_seq = self.raw_df.drop(columns = ['date', 'time', 'open']).values
        else:
            # Include 'open' in the sequence
            self.raw_seq = self.raw_df.drop(columns = ['date', 'time']).values
            # [price for tup in self.raw_df[['open', 'close']].values for price in tup]

        #self.raw_seq = np.array(self.raw_seq)
        [self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test], \
        [self.dtime_train, self.dtime_val, self.dtime_test] = \
        self._prepare_data(self.raw_seq, datetime, 
                           self.time_steps, self.forecast_steps,
                           self.batch_size, self.val_ratio, 
                           self.test_ratio, self.standardization)
        self.X = [self.X_train, self.X_val, self.X_test]
        self.y = [self.y_train, self.y_val, self.y_test]
        self.datetime = [self.dtime_train, self.dtime_val, self.dtime_test]
        print("Loading dataset... Ready for training.")
    
    def info(self):
        print("Stock Index Code: {}".format(self.index_name))
        print("Train Set Length: {}\nValidation Set Length: {}\nTest Set Length: {}"
              .format(self.y_train.shape[0],
                      self.y_val.shape[0],
                      self.y_test.shape[0]))
          
    def _prepare_data(self, raw_seq, raw_dtime, time_steps, forecast_steps,
                      batch_size, val_ratio, test_ratio, standardization):
        if standardization is not None:
            seq = self._standardize(raw_seq, standardization)
        else:
            seq = raw_seq
        
        X, y, dtime = self._build_timeser(seq, raw_dtime, time_steps, forecast_steps)
        X_train, X_test, y_train, y_test, dtime_train, dtime_test = \
        train_test_split(X, y, dtime, test_size=test_ratio, shuffle=False)
        X_train, X_val, y_train, y_val, dtime_train, dtime_val = \
        train_test_split(X_train, y_train, dtime_train, test_size=val_ratio, shuffle=False)

        seqs = [X_train, X_val, X_test, y_train, y_val, y_test]
        dtimes = [dtime_train, dtime_val, dtime_test]
        for i in range(len(seqs)):
            seqs[i] = self._trim_seq(seqs[i], batch_size)
        for i in range(len(dtimes)):
            dtimes[i] = self._trim_seq(dtimes[i], batch_size)
        
        return seqs, dtimes
    
        
    def _build_timeser(self, seq, dtime, time_steps, forecast_steps):
        dim_0 = seq.shape[0] - (time_steps + forecast_steps)
        dim_1 = seq.shape[1]
        X = np.zeros((dim_0, time_steps, dim_1))

        for i in range(dim_0):
            X[i] = seq[i : i+time_steps]
        y = seq[time_steps+forecast_steps:]
        dt = dtime[time_steps+forecast_steps:]

        return X, y, dt
    
    def _standardize(self, seq, method):
        if ('max' in method) or ('min' in method):
            scaler = MinMaxScaler()
            scaler.fit(seq)
        elif 'stand' in method:
            scaler = StandardScaler()
            scaler.fit(seq)
        seq_scaled = scaler.transform(seq)
        return seq_scaled
    
    def _trim_seq(self, seq, batch_size):
        drop = seq.shape[0] % batch_size
        if drop > 0:
            return seq[:-drop]
        else:
            return seq