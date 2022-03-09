import numpy as np
import matplotlib.pyplot as plt
import collections
from utils.utils import get_time_name
from utils.utils import categorial_vecs_to_offset_indices
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def visualize_shares(label_list, save_fig=False):
    plt.figure()

    occurrences = collections.Counter(label_list)
    labels = [x for x in list(occurrences.keys())]
    sizes = list(occurrences.values())
    # Plot
    plt.pie(sizes, labels=labels,autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')

    if save_fig:
        plt.savefig('plots/' + get_time_name() + '-pie.png')
    else:
        plt.show()

def visualize_conf(y_test, predictions, save_fig=False):
    plt.figure()

    cmdd = confusion_matrix(y_test, predictions)
    conf_md = ConfusionMatrixDisplay(confusion_matrix=cmdd)
    
    conf_md.plot()

    if save_fig:
        plt.savefig('plots/' + get_time_name() + '-conf.png')
    else:
        plt.show()