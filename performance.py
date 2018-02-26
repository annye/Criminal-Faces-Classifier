"""
Performance metrics
"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as cm


def get_confusion_matrix(X,y):
    confusion_matrix = cm(X, y)
    return confusion_matrix


def plot_confusion_matrix(cm,
                          class_names,
                          title='Confusion matrix',
                          cmap=plt.cm.autumn):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in xrange(width):
        for y in xrange(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #plt.savefig('C:\Thesis111217\CriminalClassifier\Outputs\confusion_all.png')
    #plt.clf()