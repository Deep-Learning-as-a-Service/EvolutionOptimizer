from dna.HyperparamDNA import HyperparamDNA
from dna.RandomHyperparamDNA import RandomHyperparamDNA
from evolution.EvolutionOptimizer import EvolutionOptimizer
from lib.test_fitness_func import create_test_fitness_func
from utils.utils import print_list

gen_pool = [
    ("max_depth", "natural", [0, 20]),
    ("n_estimators", "natural", [10, 200]),
    ("learning_rate", "real", [0.01, 0.4]),
    ("tree_method", "categorical", ["gpu_hist", "gpu_hist_bins", "gpu_approx", "hist", "approx"])
]

initial_individuals = [
    {'max_depth': 10, 'n_estimators': 10, 'learning_rate': 0.4, 'tree_method': 'approx'},
    {'max_depth': 0, 'n_estimators': 10, 'learning_rate': 0.4, 'tree_method': 'gpu_hist_bins'}
]

fitness = create_test_fitness_func(seconds_latency=0.2, optimal_n_estimators=100, optimal_learning_rate=0.1253875, optimal_max_depth=5, optimal_tree_method='gpu_hist')

dnas_cls_n = [
    (HyperparamDNA, 'normal'),
    (RandomHyperparamDNA, 'random')
]

best_individuals_f_n = []
best_individual_f_n_histories = []
for dna_cls_n in dnas_cls_n:
    dna = dna_cls_n[0](gen_pool)
    evo_optimizer = EvolutionOptimizer(dna.new_generation, fitness)
    best_individual_f_n, best_individual_f_n_history = evo_optimizer.fit(initial_individuals[0], initial_individuals[1], n_generations=10)
    best_individuals_f_n.append(best_individual_f_n)
    best_individual_f_n_histories.append(best_individual_f_n_history)

print("===========================================================")
print("===========================================================")
print("Two Evolutions done - COMPARISON:\n")
for i, dna_cls_n in enumerate(dnas_cls_n):
    print('======>', dna_cls_n[1], '\n')
    print("best_individuals_f_n: ", best_individuals_f_n[i], "\n")
    print("best_individual_f_n_histories:")
    print_list(best_individual_f_n_histories[i])

"""
======> normal 

best_individuals_f_n:  ({'max_depth': 20, 'n_estimators': 200, 'learning_rate': 0.112493, 'tree_method': 'gpu_hist'}, 0.9706123042162161, 'old best') 

best_individual_f_n_histories:
[
        ({'max_depth': 10, 'n_estimators': 10, 'learning_rate': 0.4, 'tree_method': 'approx'}, 7.535088464215731e-05, 'initial'),
        ({'max_depth': 10, 'n_estimators': 55, 'learning_rate': 0.256082, 'tree_method': 'gpu_hist'}, 0.0006561122514815852, 'rand 2'),
        ({'max_depth': 9, 'n_estimators': 184, 'learning_rate': 0.08777, 'tree_method': 'gpu_hist'}, 0.05976706274751138, 'rand 2'),
        ({'max_depth': 19, 'n_estimators': 190, 'learning_rate': 0.08777, 'tree_method': 'gpu_hist'}, 0.8326222841459022, 'mutate best'),
        ({'max_depth': 19, 'n_estimators': 200, 'learning_rate': 0.112493, 'tree_method': 'gpu_hist'}, 0.9567833456249953, 'mutate slightly best'),
        ({'max_depth': 20, 'n_estimators': 200, 'learning_rate': 0.10214, 'tree_method': 'gpu_hist'}, 0.9573989878552388, 'mutate slightly best'),
        ({'max_depth': 20, 'n_estimators': 200, 'learning_rate': 0.112493, 'tree_method': 'gpu_hist'}, 0.9706123042162161, 'cross 1'),
        ({'max_depth': 20, 'n_estimators': 200, 'learning_rate': 0.112493, 'tree_method': 'gpu_hist'}, 0.9706123042162161, 'old best'),
        ({'max_depth': 20, 'n_estimators': 200, 'learning_rate': 0.112493, 'tree_method': 'gpu_hist'}, 0.9706123042162161, 'old best'),
        ({'max_depth': 20, 'n_estimators': 200, 'learning_rate': 0.112493, 'tree_method': 'gpu_hist'}, 0.9706123042162161, 'old best'),
        ({'max_depth': 20, 'n_estimators': 200, 'learning_rate': 0.112493, 'tree_method': 'gpu_hist'}, 0.9706123042162161, 'old best')
]
======> random 

best_individuals_f_n:  ({'max_depth': 20, 'n_estimators': 33, 'learning_rate': 0.160121, 'tree_method': 'gpu_hist'}, 0.6734605939201299, 'old best') 

best_individual_f_n_histories:
[
        ({'max_depth': 10, 'n_estimators': 10, 'learning_rate': 0.4, 'tree_method': 'approx'}, 7.535088464215731e-05, 'initial'),
        ({'max_depth': 17, 'n_estimators': 16, 'learning_rate': 0.054813, 'tree_method': 'gpu_hist'}, 0.3516801538455913, 'rand 1'),
        ({'max_depth': 17, 'n_estimators': 16, 'learning_rate': 0.054813, 'tree_method': 'gpu_hist'}, 0.3516801538455913, 'old best'),
        ({'max_depth': 20, 'n_estimators': 21, 'learning_rate': 0.206985, 'tree_method': 'gpu_hist'}, 0.4917590905811391, 'rand 1'),
        ({'max_depth': 20, 'n_estimators': 21, 'learning_rate': 0.206985, 'tree_method': 'gpu_hist'}, 0.4917590905811391, 'old best'),
        ({'max_depth': 20, 'n_estimators': 21, 'learning_rate': 0.206985, 'tree_method': 'gpu_hist'}, 0.4917590905811391, 'old best'),
        ({'max_depth': 20, 'n_estimators': 21, 'learning_rate': 0.206985, 'tree_method': 'gpu_hist'}, 0.4917590905811391, 'old best'),
        ({'max_depth': 20, 'n_estimators': 21, 'learning_rate': 0.206985, 'tree_method': 'gpu_hist'}, 0.4917590905811391, 'old best'),
        ({'max_depth': 20, 'n_estimators': 33, 'learning_rate': 0.160121, 'tree_method': 'gpu_hist'}, 0.6734605939201299, 'rand 6'),
        ({'max_depth': 20, 'n_estimators': 33, 'learning_rate': 0.160121, 'tree_method': 'gpu_hist'}, 0.6734605939201299, 'old best'),
        ({'max_depth': 20, 'n_estimators': 33, 'learning_rate': 0.160121, 'tree_method': 'gpu_hist'}, 0.6734605939201299, 'old best')
]
"""