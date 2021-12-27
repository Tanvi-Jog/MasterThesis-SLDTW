import torch


class DataLoader:
    # the constructor loads the data
    def __init__(self, device, config):
        self.config = config
        self.trainX = torch.load(f"{self.config.data_store}/x_train_{self.config.dataset}.pt")
        self.validX = torch.load(f"{self.config.data_store}/x_valid_{self.config.dataset}.pt")
        self.testX = torch.load(f"{self.config.data_store}/x_test_{self.config.dataset}.pt")

        self.trainY = torch.load(f"{self.config.data_store}/y_train_{self.config.dataset}.pt")
        self.validY = torch.load(f"{self.config.data_store}/y_valid_{self.config.dataset}.pt")
        self.testY = torch.load(f"{self.config.data_store}/y_test_{self.config.dataset}.pt")

        print('Data Loaded.')