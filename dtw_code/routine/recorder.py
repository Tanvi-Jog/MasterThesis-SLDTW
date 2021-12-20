import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Record():
    """
    This class records and stores the loss and metric value across the training and evaluation cycles.
    Further, this class prints and plots training and metric performance.
    """
    def __init__(self, is_train):
        self.is_train = is_train
        self.name = "TRAIN" if is_train else "TEST"
        self.loss = list()

    def update(self, idx, loss):
        self.loss.append(loss)
        #calculate the metrics here
        #output, y = np.array(output), np.array(y)
        print("Epoch: %d, %s loss: %1.5f" % (idx, self.name, loss))
        print("=" * 60)

    def get_lowest_loss(self):
        return round(min(self.loss), 4)

    def plot_losses(self, loss_values, title):
        if not self.is_train:
            plt.figure(figsize=(12, 7))
            plt.grid()
            plt.plot([x for x in self.loss], 'go--', linewidth=2, markersize=5, label='Test Loss')
            plt.plot([x for x in loss_values], 'r*-', linewidth=2, markersize=5, label='Train Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss Values')
            plt.title(title)
            image_name = f'{title.replace(" ", "_").replace(":", "_")}_{time.strftime("%d_%m_%H_%M")}.png'
            plt.legend()
            plt.savefig(os.path.join("./routine", "plots", image_name))
        else:
            print("WARNING: CANNOT INVOKE LOSS PLOT")


