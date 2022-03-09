"""
Simple Multi-Layer Dense-Net Evolution Optimization for recognizing diabetes cases
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
from evolution.EvolutionOptimizer import EvolutionOptimizer
from dna.SimpleDeepNetDNA import SimpleDeepNetDNA


def simple_deep_net_evolution():
    print("\nStart simple_deep_net_evolution() --------------------------------------------------")

    # fitness function preparation
    dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=',')
    X = dataset[:,0:8]
    y = dataset[:,8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def create_model(individual):
        model = Sequential() # keras.models.Sequential()
        for idx, layer_size in enumerate(individual):
            if idx == 0:
                model.add(Dense(layer_size, input_dim=8, activation='relu')) # Input layer
            else:
                model.add(Dense(layer_size, activation='relu'))
        
        model.add(Dense(1, activation='sigmoid')) # Output
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def get_k_fold_mean_accuracy(model_create_fit_evalutate, X_train, y_train, k=3):
        kf = KFold(n_splits=k, shuffle=True, random_state=2)
        accuracies = []
        for train_index, test_index in kf.split(X_train):
            X_val_train, X_val_test = X_train[train_index], X_train[test_index]
            y_val_train, y_val_test = y_train[train_index], y_train[test_index]
            _, accuracy = model_create_fit_evalutate(X_val_train, y_val_train, X_val_test, y_val_test)
            accuracies.append(accuracy)

        return np.mean(accuracies)  
    
    def model_create_fit_evalutate_func(individual):
        def model_create_fit_evalutate(X_train, y_train, X_test, y_test):
            verbose = 0
            model = create_model(individual)
            model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=verbose) # verbose=1 looks nice
            return model, model.evaluate(X_test, y_test, verbose=verbose)[1]
        return model_create_fit_evalutate
    
    print("\nEvolution --------------------------------------------------")

    def fitness(individual):
        model_create_fit_evalutate = model_create_fit_evalutate_func(individual)
        return get_k_fold_mean_accuracy(model_create_fit_evalutate, X_train, y_train, k=4)
    
    gen_pool = {
        'n_layers_range': [1, 10], # net meta var
        'units': [1, 64], # n_neurons_per_layer_range
    }

    dna = SimpleDeepNetDNA(gen_pool)

    # # check quality of fitness function
    
    # random_individuals = dna.get_random_individuals(10)
    # for individual in random_individuals:
    #     model_create_fit_evalutate = model_create_fit_evalutate_func(individual)
    #     _, accuracy = model_create_fit_evalutate(X_train, y_train, X_test, y_test)

    #     print("individual:", individual, "kfold fitness 1:", fitness(individual), "kfold fitness 2:", fitness(individual), "final fitness:", accuracy, "\n\n")

    evo_optimizer = EvolutionOptimizer(dna.new_generation, fitness)
    
    initial_individuals = [
        [14], 
        [18, 21, 17],
    ]

    best_individual_f_n, best_individual_f_n_history = evo_optimizer.fit(initial_individuals[0], initial_individuals[1], n_generations=10)
    

    print("\nReal Evaluation of best_individual: ", best_individual_f_n ," --------------------------------------------------")
    model_create_fit_evalutate = model_create_fit_evalutate_func(best_individual_f_n[0])
    model, accuracy = model_create_fit_evalutate(X_train, y_train, X_test, y_test)
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    print("\nFinal accuracy: ", accuracy)
    predictions = (model.predict(X_test) > 0.5).astype(int)
    # same as: accuracy_score(y_test, predictions)
    visualize_shares(y_test, save_fig=True)
    visualize_conf(y_test, predictions, save_fig=True)

# simple_deep_net_evolution()

"""
Problem:
    - fitness function is not consitent, if you call it twice there could be a diffence of up to 5 percent!
    - difference between fitness four-fold and the final fitness!

- best individual history:

[
        ([22, 15, 15], 0.7159035801887512, 'initial'),
        ([15, 22, 20], 0.7352077215909958, 'cross between 1'),
        ([15, 15, 15], 0.7353591322898865, 'cross 1'),
        ([15, 15, 15], 0.7353591322898865, 'old best'),
        ([17, 21, 20], 0.7490007281303406, 'cross between 1'),
        ([17, 21, 20], 0.7490007281303406, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'cross between 1'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best'),
        ([18, 21, 17], 0.7547389715909958, 'old best')
]


=======================================


Real Evaluation of best_individual:  ([18, 21, 17], 0.7547389715909958, 'old best')  --------------------------------------------------

Final accuracy:  0.7204724550247192

"""

