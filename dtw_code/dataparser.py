import numpy as np
import pandas as pd
import torch

from tqdm.notebook import tqdm
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


class DataParser:
    # the constructor loads the data
    def __init__(self, data_path, config):
        self.config = config
        self.window_width = self.config.window
        self.future_steps = self.config.horizon
        # load the data
        data = self.load_data(data_path, [self.config.data_period], self.config.dataset)
        # taking transpose of the data
        data_t = data.T
        print(data_t.shape)
        total_data = np.array([data_t.loc[i,:] for i in data_t.index])
        print('Total data is of shape: ', total_data.shape)
        print('Implementing sliding window...')
        total_X, total_Y = self.generate_data(total_data, self.window_width, self.future_steps)
        print('X shape:', total_X.shape,' Y shape:', total_Y.shape)
    
        # split the data into train-valid-test
        trainX, testX, trainY, testY = train_test_split(total_X, total_Y, test_size=0.2, shuffle=False, random_state=11)
        trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.2, shuffle=False, random_state=11)
      
        # augumenting train data by adding 20% data
        # trainX, trainY = self.data_augument(trainX, trainY)

        # normalizing the data
        scaler = self.get_scaler('standard')
        trainX_arr = scaler.fit_transform(trainX)
        validX_arr = scaler.transform(validX)
        testX_arr = scaler.transform(testX)

        trainY_arr = scaler.fit_transform(trainY)
        validY_arr = scaler.transform(validY)
        testY_arr = scaler.transform(testY)

        trainX = torch.Tensor(trainX_arr)
        trainY = torch.Tensor(trainY_arr)
        validX = torch.Tensor(validX_arr)
        validY = torch.Tensor(validY_arr)
        testX = torch.Tensor(testX_arr)
        testY = torch.Tensor(testY_arr)

        print('Data shapes are:')
        print('TrainX: ', trainX.shape, 'validX: ', validX.shape, 'testX: ', testX.shape)
        print('TrainY: ', trainY.shape, 'validY: ', validY.shape, 'testY: ', testY.shape)

        print('Storing the data...')
        # features
        torch.save(trainX, f"{self.config.data_store}/x_train_{self.config.dataset}.pt")
        torch.save(validX, f"{self.config.data_store}/x_valid_{self.config.dataset}.pt")
        torch.save(testX, f"{self.config.data_store}/x_test_{self.config.dataset}.pt")
        # targets
        torch.save(trainY, f"{self.config.data_store}/y_train_{self.config.dataset}.pt")
        torch.save(validY, f"{self.config.data_store}/y_valid_{self.config.dataset}.pt")
        torch.save(testY, f"{self.config.data_store}/y_test_{self.config.dataset}.pt")

        print('Data preprocessing completed.')

    def load_data(self, data_path, data_period, dataset):
      if dataset == 'electricity':
        df = pd.read_csv(data_path, sep=';', decimal=',')
        df.rename(columns = {"Unnamed: 0": "time"}, inplace = True)
        df['time'] = pd.to_datetime(df['time'])
        df_included = df[df["time"].dt.year.isin(data_period)].reset_index()
        df_included.drop('index', axis=1, inplace=True)
        # hourly level aggregation
        df_included.index = pd.to_datetime(df_included['time'])
        df_included.sort_index(inplace=True)
        df_included = df_included.resample('1h').mean()

      if dataset == 'cost_data':
        df = pd.read_csv(data_path)
        df = df.drop(['Unnamed: 0'], axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_included = df[df["timestamp"].dt.year.isin(data_period)]
        df_included = df_included.set_index('timestamp')
        
      return df_included

    # function to convert data into x,y for supervised learning
    def generate_data(self, total_data, seq_length, pred_window):
      x = []
      y = []
      for idx in tqdm(range(len(total_data))):
        data = total_data[idx]
        for i in range(len(data)-seq_length):
          _x = data[i:(i+seq_length)]
          _y = data[i+seq_length:i+seq_length+pred_window]
          if len(_y) < pred_window:
            continue
          else:
            x.append(_x)
            y.append(_y)

      dataX = Variable(torch.Tensor(x))
      dataY = Variable(torch.Tensor(y))
      return dataX, dataY

    # funtion to split the dataset into train/valid/test
    def _split(self, train, valid, test, n, P, m, h):
      train_set = range(P+h-1, train)
      valid_set = range(train, valid)
      test_set = range(valid, n)
      return train_set, valid_set, test_set

    # function to get the scaler for normalization
    def get_scaler(self, scaler):
        scalers = {
          "minmax": MinMaxScaler,
          "standard": StandardScaler
        }
        return scalers.get(scaler.lower())()

    def data_augument(self, trainX, trainY):
      n, m = trainX.shape
      noise_size = abs(int(n*self.config.train_size) - int(n*0.7)) 
      print('Adding noise of size:', int(noise_size))
      noise_X = np.random.normal(0, .1, trainX[0].shape)
      noise_Y = np.random.normal(0, .1, trainY[0].shape)
      t_X = []
      t_Y = []
      for i in tqdm(range(int(noise_size))):
        train_rand_row_idx = np.random.randint(0, trainX.shape[0])
        # print(train_rand_row_idx)
        noise_row_X = trainX[train_rand_row_idx].numpy() + noise_X
        noise_row_Y = trainY[train_rand_row_idx].numpy() + noise_Y
        t_X.append(noise_row_X)
        t_Y .append(noise_row_Y)
      t_X = torch.Tensor(t_X)
      t_Y = torch.Tensor(t_Y)
      trainX_aug = torch.concat((trainX, torch.Tensor(t_X)))
      trainY_aug = torch.concat((trainY, torch.Tensor(t_Y)))
      return trainX_aug, trainY_aug