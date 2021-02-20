import matplotlib.pyplot as plt

def plot_history(history, metric):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history[metric])
    plt.plot(history['val_' + metric])
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc=0)
    plt.show()