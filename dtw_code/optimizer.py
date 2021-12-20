import torch
import torch.nn as nn
import numpy as np

from tslearn.metrics import dtw, dtw_path
from torch.utils.data import DataLoader, Dataset, TensorDataset
from routine.recorder import Record

class Optimizer:

    def __init__(self, config, dataset, model_pred, model_loss):
      self.config = config
      self.ds = dataset
      self.model_pred = model_pred
      self.model_loss = model_loss

      train_data = TensorDataset(self.ds.trainX, self.ds.trainY)
      self.train_loader = DataLoader(train_data, shuffle=True, batch_size=self.config.batch_size, drop_last=True)
      val_data = TensorDataset(self.ds.validX, self.ds.validY)
      self.val_loader = DataLoader(val_data, shuffle=False, batch_size=self.config.batch_size, drop_last=True)
      test_data = TensorDataset(self.ds.testX, self.ds.testY)
      self.test_loader = DataLoader(test_data, shuffle=False, batch_size=self.config.batch_size, drop_last=True)

    def run(self):
        best_loss = 0
        #pretrain the prediction network
        if str(self.config.pretrain_pred).lower() == 'true':
          print('pretraining prediction network')
          self.train_prediction(self.config.epochs, 'mse', self.train_loader, self.val_loader)
        else:
          print('Loading the pretrained prediction network')
          self.model_pred.load_state_dict(torch.load(f"{self.config.save}/{self.config.dataset}_prediction_model.pth"))
        #train the surrogate network
        if str(self.config.pretrain_loss).lower() == 'true':
          print('pretraining surrogate network')
          self.train_surrogate(self.config.epochs, self.val_loader, self.test_loader)
        else:
          print('Loading the pretrained surrogate network')
          self.model_loss.load_state_dict(torch.load(f"{self.config.save}/{self.config.dataset}_surrogate_model.pth"))
        
        for epoch in range(self.config.epochs):
            #train prediction network for surrogate loss
            self.train_prediction(self.config.steps_prediction, 'dtw_surrogate', self.train_loader, self.val_loader)
            #train surrogate network again
            self.train_surrogate(self.config.steps_loss, self.train_loader, self.val_loader)
            #compute true dtw loss
            L_true = self.epoch_func(self.model_loss, self.model_pred, self.test_loader, self.config.batch_size, n_features=1, 
                                     train_model='prediction', loss_type='dtw')
            #compute surrogate dtw loss
            L_hat = self.epoch_func(self.model_loss, self.model_pred, self.test_loader, self.config.batch_size, n_features=1, 
                                    train_model='surrogate', loss_type='dtw')
            if (L_true-L_hat) < best_loss:
                best_loss = (L_true-L_hat)
                print("Epoch: %d, L_true: %1.5f, L_hat: %1.5f" % (epoch, L_true, L_hat))
                #save the corresponding prediction and surrogate models
                torch.save(self.model_pred.state_dict(), f"best_models/prediction_net_{self.config.dataset}.pth")
                torch.save(self.model_loss.state_dict(), f"best_models/surrogate_net_{self.config.dataset}.pth")
            print("*"*60)

    def train_prediction(self, num_epochs, type_loss, train_dataset, test_dataset):
        '''
        Train the forecasting model with Euclidean loss 
        '''
        loss =  {'train loss':list(), 'valid loss':list(), 'test loss':list()} 
        title = f"{self.config.dataset}_prediction_model"
        best_valid_loss = float("inf")
        optimizer = torch.optim.Adam(self.model_pred.parameters(),lr=self.config.lr_pred, weight_decay=1e-6)
            
        for epoch in range(num_epochs): 
            train_loss = self.train(self.model_loss, self.model_pred, train_dataset, self.config.batch_size, n_features=1, 
                                    train_model='prediction', loss_type=type_loss, opt=optimizer)
            valid_loss = self.test(self.model_loss, self.model_pred, test_dataset, self.config.batch_size, n_features=1, 
                                    train_model='prediction', loss_type=type_loss)
            loss['train loss'].append(train_loss)
            loss['valid loss'].append(valid_loss)
            print("Epoch: %d, Train loss: %1.5f, Valid loss: %1.5f" % (epoch, train_loss, valid_loss))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #save and load only the model parameters
                torch.save(self.model_pred.state_dict(), f"{self.config.save}/{title}.pth")
                es = 0
            else:
                es += 1
                print("Counter {} of 30".format(es))
                if es > 30:
                    print("Early stopping with best_validation_loss: ", best_valid_loss)
                    break
            print("="*60)
        self.model_pred.load_state_dict(torch.load(f"{self.config.save}/{title}.pth"))
            
    
    def train_surrogate(self, num_epochs, train_dataset, test_dataset):
        '''
        Training the surrogate loss model.
        '''
        loss =  {'train loss':list(), 'valid loss':list(), 'test loss':list()}
        title = f"{self.config.dataset}_surrogate_model"
        best_valid_loss = float("inf")
        optimizer = torch.optim.Adam(self.model_loss.parameters(),lr=self.config.lr_loss, weight_decay=1e-6)

        for epoch in range(num_epochs):
            train_loss = self.train(self.model_loss, self.model_pred, train_dataset, self.config.batch_size, n_features=1, 
                                         train_model='surrogate', loss_type='dtw_surrogate', opt=optimizer)
            valid_loss = self.test(self.model_loss, self.model_pred, test_dataset, self.config.batch_size, n_features=1, 
                                         train_model='surrogate', loss_type='dtw_surrogate')
            loss['train loss'].append(train_loss)
            loss['valid loss'].append(valid_loss)
            print("Epoch: %d, Train loss: %1.5f, Valid loss: %1.5f" % (epoch, train_loss, valid_loss))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #save and load only the model parameters
                torch.save(self.model_pred.state_dict(), f"{self.config.save}/{title}.pth")
                es = 0
            else:
                es += 1
                print("Counter {} of 30".format(es))
                if es > 30:
                    print("Early stopping with best_validation_loss: ", best_valid_loss)
                    break
            print("="*60)
        self.model_loss.load_state_dict(torch.load(f"{self.config.save}/{title}.pth"))

    
    def train(self, model_loss, model_pred, data_loader, batch_size, n_features, train_model, loss_type=None, opt=None):
        '''
        Function to carry out training for each epoch
        '''
        total_loss = 0.
        if train_model == 'prediction':
            model_pred.train()
        if train_model == 'surrogate':
            model_loss.train()

        for i, data in enumerate(data_loader, 0):
            loss_dtw = 0
            x, y = data
            x = x.view([batch_size, -1, n_features])
            if torch.cuda.is_available():
                x = torch.tensor(x, dtype=torch.float32).cuda()
                y = torch.tensor(y, dtype=torch.float32).cuda()
            outputs = model_pred(x)
            #euclidean loss
            if loss_type == 'mse':
                criterion = torch.nn.MSELoss()
                loss = criterion(outputs, y)
                total_loss += loss.item()
            #surrogate loss
            if loss_type == 'dtw_surrogate':
                y = y.view([batch_size, -1, n_features])
                y_hat = outputs.view([batch_size, -1, n_features])
                loss = model_loss(y, y_hat)
                total_loss += loss
            #true dtw loss 
            if loss_type == 'dtw':
                y = y.view([batch_size, -1, n_features])
                y_hat = outputs.view([batch_size, -1, n_features])
                if torch.cuda.is_available():
                    y_hat = torch.tensor(y_hat, dtype=torch.float32).cuda()
                for k in range(self.config.batch_size):
                    target_k_cpu = y[k,:,0:1].view(-1).detach().cpu().numpy()
                    output_k_cpu = y_hat[k,:,0:1].view(-1).detach().cpu().numpy()
                    path, sim = dtw_path(target_k_cpu, output_k_cpu)
                    loss_dtw += sim
                loss_dtw = loss_dtw /batch_size
                total_loss += loss_dtw
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss /= len(data_loader.dataset)
        return total_loss

    def test(self, model_loss, model_pred, data_loader, batch_size, n_features, train_model, loss_type=None, opt=None):
        '''
        Function to carry out testing for each epoch
        '''
        total_loss = 0.
        if train_model == 'prediction':
            model_pred.eval()
        if train_model == 'surrogate':
            model_loss.eval()

        for i, data in enumerate(data_loader, 0):
            loss_dtw = 0
            x, y = data
            x = x.view([batch_size, -1, n_features])
            if torch.cuda.is_available():
                x = torch.tensor(x, dtype=torch.float32).cuda()
                y = torch.tensor(y, dtype=torch.float32).cuda()
            outputs = model_pred(x)
            #euclidean loss
            if loss_type == 'mse':
                criterion = torch.nn.MSELoss()
                loss = criterion(outputs, y)
                total_loss += loss.item()
            #surrogate loss
            if loss_type == 'dtw_surrogate':
                y = y.view([batch_size, -1, n_features])
                y_hat = outputs.view([batch_size, -1, n_features])
                loss = model_loss(y, y_hat)
                total_loss += loss
            #true dtw loss 
            if loss_type == 'dtw':
                y = y.view([batch_size, -1, n_features])
                y_hat = outputs.view([batch_size, -1, n_features])
                if torch.cuda.is_available():
                    y_hat = torch.tensor(y_hat, dtype=torch.float32).cuda()
                for k in range(self.config.batch_size):
                    target_k_cpu = y[k,:,0:1].view(-1).detach().cpu().numpy()
                    output_k_cpu = y_hat[k,:,0:1].view(-1).detach().cpu().numpy()
                    path, sim = dtw_path(target_k_cpu, output_k_cpu)
                    loss_dtw += sim
                loss_dtw = loss_dtw /batch_size
                total_loss += loss_dtw
        total_loss /= len(data_loader.dataset)
        return total_loss