import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool1d

class SurrogateModel(nn.Module):
  """
  This model extracts the features needed for similarity from inputs. The architecture replicates the best performing
  architecture. HP for the Siamese are fixed and must be avoided for replicating the original results.
  """
  def __init__(self, config): #future_steps, batch_size, 1
    super(SurrogateModel, self).__init__()
    self.config = config
    self.input_dim = self.config.horizon
    self.hidden_dim = self.config.batch_size
    self.output_dim = 1
    self.conv = nn.Sequential(
        nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1),
        nn.Conv1d(in_channels=self.hidden_dim, out_channels=64, kernel_size=1, dilation=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
        nn.BatchNorm1d(32),
        nn.ReLU()
    )
    self.fc1 = nn.Linear(32, 16)
    self.fcOut = nn.Linear(16, 1)
    #trial
    self.conv_layer = nn.Sequential(
      #layer 1
      nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=1),
      #layer 2
      nn.Conv1d(in_channels=self.hidden_dim, out_channels=64, kernel_size=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=1),
      #layer 3
      nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=1),
      #layer 2
      nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=1),
    )
    self.fc1_layer = nn.Linear(16, 8)
    self.fcOut_layer = nn.Linear(16, 1)

  def forward(self, x1, x2):
    """
    This is invoked during training. The output is real value depicting whether
    the two signals are similar or different based on the absolute difference in
    the feature extracted from the conv module.
    :param x1: Signal 1 aka targets
    :param x2: Signal 2 aka predictions
    :return: Real value score
    """
    x1 = self.conv(x1)
    x1 = x1.reshape(x1.size(0), -1)
    x1 = self.fc1(x1)
    x1 = nn.Sigmoid()(x1)
    x2 = self.conv(x2)
    x2 = x2.reshape(x2.size(0), -1)
    x2 = self.fc1(x2)
    x2 = nn.Sigmoid()(x2)
    x = torch.abs(x1 - x2)
    x = self.fcOut(x)
    x = (torch.sum(x)) #or nn.Sigmoid() torch.sum(x)
    return x

