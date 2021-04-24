import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_history(history, out_dir):

    plt.plot(history.history['acc'], label='training')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(out_dir, "acc.png"), dpi=300, bbox_inches='tight')
    plt.clf()
  
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=300, bbox_inches='tight')
    plt.clf()
  