"""
Simple Mulitlayer Dense Neural Network for for diabetis prediction
- parameterize sequential models of this kind and compare accuracies
"""


# first neural network with keras make predictions
from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.visualization import visualize_shares, visualize_conf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from keras.utils.vis_utils import plot_model
import keras.backend as K


def create_model(individual):
    """
    recommended model:
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    - input dims need to be 8, because of the data
    - the output layer need to be 1 (binary classification)

    Evolution on individual: (12, 8) -> 2 Dense layer with 12 and 8 neurons (recommended model)
    other example: (5, 6, 7, 5) -> 4 Dense layer
    - at least one layer

    map it to accuracy and improve it!
    later add the activation function: [(5, 'relu'), (8, 'sigmoid)] as one individual, also general hyperparameters: loss, optimizer, epochs, batch_size
    """

    model = Sequential() # keras.models.Sequential()
    for idx, layer_size in enumerate(individual):
        if idx == 0:
            model.add(Dense(layer_size, input_dim=8, activation='relu')) # Input layer
        else:
            model.add(Dense(layer_size, activation='relu'))
    
    model.add(Dense(1, activation='sigmoid')) # Output
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)    

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 12)                108       
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 104       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 9         
=================================================================
Total params: 221
Trainable params: 221
Non-trainable params: 0
_________________________________________________________________
"""

def get_k_fold_mean_accuracy(model_create_fit_evalutate, X_train, y_train, k=3):
    print("\nKFold Evaluation --------------------------------------------------")
    kf = KFold(n_splits=k, shuffle=True, random_state=2)
    accuracies = []
    for train_index, test_index in kf.split(X_train):
        X_val_train, X_val_test = X_train[train_index], X_train[test_index]
        y_val_train, y_val_test = y_train[train_index], y_train[test_index]
        _, accuracy = model_create_fit_evalutate(X_val_train, y_val_train, X_val_test, y_val_test)
        accuracies.append(accuracy)

    
    print("\naccuracies: ", accuracies)
    print("mean accuracy: ", np.mean(accuracies))


# MAIN --------------------------------------------------------------
# load the dataset
dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

individual = [0]
net_description = "8-"
for layer_size in individual:
    net_description += str(layer_size) + "-"
net_description += "1"
print("\nStart --------------------------------------------------")
print("\n- will test model with net_description: ", net_description, " (individual: ", individual, ")\n")

def model_create_fit_evalutate(X_train, y_train, X_test, y_test):
    model = create_model(individual)
    print("Start fitting")
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0) # verbose=1 looks nice
    print("End fitting")
    return model, model.evaluate(X_test, y_test)[1]

get_k_fold_mean_accuracy(model_create_fit_evalutate, X_train, y_train, k=3)

print("\nReal Evaluation --------------------------------------------------")
model, accuracy = model_create_fit_evalutate(X_train, y_train, X_test, y_test)
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print("\nFinal accuracy: ", accuracy)
predictions = (model.predict(X_test) > 0.5).astype(int)
# same as: accuracy_score(y_test, predictions)
visualize_shares(y_test, save_fig=True)
visualize_conf(y_test, predictions, save_fig=True)
