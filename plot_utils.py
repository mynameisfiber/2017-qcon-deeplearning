import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import keras.backend as K

import itertools
from functools import reduce


def plot_digit(images, prediction, n_digits=16):
    assert n_digits % 4 == 0
    images_and_predictions = list(zip(images, prediction))
    for index, (image, prediction) in enumerate(images_and_predictions[:n_digits]):
        plt.subplot(n_digits/2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)
    plt.show()
    
def plot_train_history(histories):
    fields = reduce(set.union, (set(h.history.keys()) for h in histories.values()))
    subplots = plt.subplots(len(fields))
    max_epoch = max(max(h.epoch) for h in histories.values())
    for key, ax in zip(fields, subplots[1]):
        ax.set_ylabel(key)
        ax.set_xlabel("Epoch")
        for name, history in histories.items():
            if key in history.history:
                ax.plot(history.epoch, history.history[key], label=name)
        ax.set_xlim(xmin=0, xmax=max_epoch)
    plt.legend()
    plt.show()

# confusion matrix plotting routine from:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def _get_activations(model, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                 [layer.output for layer in model.layers])
    activations = get_activations([X_batch,0])
    return zip(model.layers, activations)
    
def plot_activations(model, X):
    activations = list(_get_activations(model, [X]))
    print([a[1].shape for a in activations])
    for layer, activation in activations[1:]:
        if len(activation.shape) == 4:
            N = activation.shape[1]
            if N == 1:
                plt.imshow(activation[0, 0, :, :])
                plt.axis('off')
            else:
                fig, axes = plt.subplots(ncols=N)
                for i, ax in enumerate(axes):
                    ax.imshow(activation[0, i, :, :])
                    ax.axis('off')
        else:
            N = activation.shape[-1]
            y = 4
            x = int(N / y)
            while x*y != N and y < N:
                y += 1
                x = int(N/16)
            if x*y == activation.shape[-1]:
                act = activation[0, :].reshape((y, x))
                if layer.name != 'output':
                    plt.axis('off')
            else:
                act = activation
            plt.imshow(act)
        plt.title(layer.name)
        plt.show()