import matplotlib.pyplot as plt
import numpy as np

def plot_lstm_validation_curve_epochs():
    """Plots the impact of the number of epochs on the LSTM validation curve.
       Saves plot in current folder."""
    plt.clf()
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.282, 0.5)
    plt.yticks(np.arange(0.27, 0.5, step=0.025))
    plt.xlim(1, 5)
    lw = 2
    plot_fn = plt.plot
    param_range = [1, 2, 3, 4, 5]
    plot_fn(param_range, [0.2841, 0.3663, 0.4015, 0.4296, 0.4542], label='Training accuracy',
            color='r', marker='^', lw=lw)
    plot_fn(param_range, [0.3344, 0.3642, 0.4004, 0.4091, 0.4185], label=' Validation accuracy',
            color='b', marker='^',  lw=lw)
    plt.legend(loc='best')
    plt.savefig('lstm_validation_curve_epochs.png')
    return plt


if __name__ == "__main__":
    plot_lstm_validation_curve_epochs()