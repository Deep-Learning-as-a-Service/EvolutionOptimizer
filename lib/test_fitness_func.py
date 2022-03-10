import time
import numpy as np

def create_test_fitness_func(seconds_latency=0.5, optimal_n_estimators=100, optimal_learning_rate=0.1253875, optimal_max_depth=5, optimal_tree_method='gpu_hist'):
    def test_fitness_func(individual):
        fitness = -20 # sigmoid between -6 and 6

        # learning_rate
        distance = abs(optimal_learning_rate - individual['learning_rate'])
        fitness += ((1 - distance)**5)*8 # can come to +6 -> 0 -> simoid 50 percent
        
        # n_estimators
        max_distance = 100
        distance = abs(optimal_n_estimators - individual['n_estimators']) # between 0 and 100
        fitness += 6 * (distance / max_distance) # can get max 3 improvement

        # max_depth
        max_distance = 15
        distance = abs(optimal_max_depth - individual['max_depth']) # between 0 and 15
        fitness += 6 * (distance / max_distance) # can get max 3 improvement

        # tree_method
        tree_methods = ['gpu_hist', 'gpu_hist_bins', 'gpu_approx', 'hist', 'approx']
        tree_methods.remove(optimal_tree_method)
        if individual['tree_method'] == optimal_tree_method:
            fitness += 4
        else:
            fitness += (tree_methods.index(individual['tree_method']) / 2)

        time.sleep(seconds_latency)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        return sigmoid(fitness)
        
    return test_fitness_func

