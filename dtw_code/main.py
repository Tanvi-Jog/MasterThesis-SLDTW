import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

from surrogate_model import SurrogateModel
from prediciton_model import PredictionModel
from new_optimizer import Optimizer
from dataparser import DataParser
from dataloader import DataLoader

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting and dtw surrogate loss')
parser.add_argument('--dataset', type=str, required=True, help='name of the dataset')
parser.add_argument('--data', type=str, required=True, help='location of the data file')
parser.add_argument('--data_process', type=str, default=False, help='location of the data file')
parser.add_argument('--data_store', type=str, default='stored_features', help='location of the data file')
parser.add_argument('--data_period', type=int, default=2021, help='year for the data to be included')
parser.add_argument('--window', type=int, default=24 * 7, help='window size')
parser.add_argument('--horizon', type=int, default=12, help='future steps to be predicted')
parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--save', type=str,  default='stored_model', help='path to save the final model')
parser.add_argument('--lr_pred', type=float, default=0.01, help='learning rate for prediction model')
parser.add_argument('--lr_loss', type=float, default=0.01, help='learning rate for surrogate loss model')
parser.add_argument('--steps_loss', type=int, default=3, help='the number of mini-batch update steps for learning the '
                                                              'loss model parameters')
parser.add_argument('--steps_prediction', type=int, default=10, help='the number of mini-batch update steps '
                                                                     'for learning')
parser.add_argument('--loss_type', type=str,  default='dtw', help='default surrogate loss for')
parser.add_argument('--pretrain_pred', type=str, default=False)
parser.add_argument('--pretrain_loss', type=str, default=False)
parser.add_argument('--train_size', type=float, default=0.5, help='portion of dataset for training')
parser.add_argument('--valid_size', type=float, default=0.2, help='portion of dataset for validation')
parser.add_argument('--test_size', type=float, default=0.3, help='portion of dataset for testing')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available device is: ', device, torch.cuda.device_count())

# load, parse and store the data only if indicated o/w load the processed data.
if str(args.data_process).lower() == 'true':
    data = DataParser(data_path=args.data, config=args)

# read the dataset
ds = DataLoader(device, config=args)

# create the prediction model
model_pred = PredictionModel(config=args).to(device)

# create the surrogate model
model_loss = SurrogateModel(config=args).to(device)

# create the optimizer
optimizer = Optimizer(config=args, dataset=ds, model_pred=model_pred, model_loss=model_loss)
optimizer.run()
