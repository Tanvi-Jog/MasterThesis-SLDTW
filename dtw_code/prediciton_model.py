import torch
import torch.nn as nn

class PredictionModel(nn.Module):
  """
  This model does time series forecasting for given sequence length and forecast horizon. The architecture replicates the best performing
  architecture. HP for the CNN are fixed and must be avoided for replicating the original results.
  """
  def __init__(self, config): #window_width, batch_size, future_steps
    super(PredictionModel, self).__init__()
    self.config = config
    self.input_dim = self.config.window
    self.hidden_dim = self.config.batch_size
    self.output_dim = self.config.horizon
    self.layer = nn.Sequential(
        nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1),
        # nn.MaxPool1d(kernel_size=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=self.hidden_dim, out_features=10),
        nn.Linear(10, self.output_dim)
    )
    self.conv_layer = nn.Sequential(
      nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1),
      nn.Conv1d(in_channels=self.hidden_dim, out_channels=16, kernel_size=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=1),
      nn.Flatten(),
      nn.Linear(in_features=16, out_features=10),
      nn.Linear(10, self.output_dim)
    )
  def forward(self, x):
    outputs = self.layer(x)
    return outputs


'''
self.layer = nn.Sequential(
        nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1),
        nn.ReLU(),
        # nn.MaxPool1d(kernel_size=1),
        # nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(in_features=self.hidden_dim, out_features=50),
        nn.Linear(50, self.output_dim)
    )
'''