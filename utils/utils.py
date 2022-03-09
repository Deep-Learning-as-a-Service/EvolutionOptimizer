import os
import tensorflow as tf
import numpy as np
import random
import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split, LeaveOneGroupOut
# from tensorflow_addons.metrics import F1Score
from sklearn.metrics import f1_score



def get_activity_indices():
    activityIndexLists = { x: [] for x in activityMap.keys()}
    for it, actIndex in enumerate(activityMap.keys()):
        actIndex = int(actIndex)
        #print(it, actIndex)
        activityIndexLists[actIndex].append(it)
    
    for key, indices in activityIndexLists.items():
        if len(indices) == 0:
            print(print('Didnt find activity ', i))
    
    return activityIndexLists

# Generate stable results for submitting
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(2)
   tf.random.set_seed(2)
   np.random.seed(2)
   random.seed(2)

def get_time_name() -> str:
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")

def three_shuffle(subjects, activities, windows) -> None:
    joined = list(zip(subjects, activities, windows))
    random.shuffle(joined)
    subjects, activities, windows = zip(*joined)
    return np.array(subjects), np.array(activities), np.array(windows)

def categorial_vecs_to_offset_indices(vecs: np.ndarray) -> np.ndarray:
    return np.argmax(vecs, axis=1) + 1

def write_txt_between(turn_on=False):
    if turn_on:
        file = open("plots/" + get_time_name() + "---------------------------------.txt", "w") 
        file.write("-------") 
        file.close()

def calculate_f1_for_validation(y_test, predictions):
    '''
    Returns an array with a score for each class
    for the challenge the mean is used
    '''
    return np.mean(f1_score(categorial_vecs_to_offset_indices(y_test), categorial_vecs_to_offset_indices(predictions), average='macro'))
    # metric = F1Score(num_classes=10)
    # metric.update_state()
    # return metric.result()

def leave_one_out_evaluation(X_all, y_all, subjects):
    splitter = LeaveOneGroupOut()
    return splitter.split(X_all, y_all, subjects)

def k_fold_evaluation(X_all, y_all, k=4):
    splitter = KFold(n_splits=k, random_state=None)
    return splitter.split(X_all, y_all)
    
def print_list(li, print_func=print):
    print_func('[\n\t' + ',\n\t'.join(list(map(str, li))) + '\n]')

def dicts_equal_n_exceptions(dict_1, dict_2, n_exceptions):
    for key, value in dict_1.items():
        if dict_2[key] != value:
            if n_exceptions == 0:
                return False
            n_exceptions -= 1
    return True

