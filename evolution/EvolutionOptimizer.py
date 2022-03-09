import random
import numpy as np
from utils.progress_bar import printProgressBar
from utils.utils import print_list
import time
import json
from utils.telegram import send_telegram
from dna.XGBHyperparamDNA import XGBHyperparamDNA

class EvolutionOptimizer():
    def __init__(self, new_generation, fitness, verbose=True, log_func=print):
        """
            - individual_1 == individual_2 needs to work
            :new_generation: (individual_1, individual_2) -> (individual_n_1, individual_n_2)
            :fitness: (individual) -> fitness (float)

            :verbose: bool
            :log_func: (*args, **kwargs) -> None
        """
        # Evolution Config: Individual
        self.new_generation = new_generation
        self.fitness = fitness

        # Evolution Quality
        self.n_calls_fitness = 0
        self.fitness_cache = ([], []) # (individuals, fitnesses)
        # O(n)... because dicts/lists (possible instances) are not hashable!
        # change to dict if Individuals get hashable types and the comparring works????
        self.fitness_cache_hits = 0
        self.n_duplicates_in_gens = []
        self.n_fitness_duplicates_in_gens = []

        # Logging
        self.verbose_print = log_func if verbose else lambda *a, **k: None
        progress_bar = lambda prefix, suffix, progress, total: printProgressBar(progress, total, prefix = prefix, suffix = suffix, length = 30, log_func = self.verbose_print)
        self.progress_bar_fitting = lambda prefix, progress, total: progress_bar(prefix, '- ' + str(progress) + '/' + str(total) + ' fitted', progress, total)
        self.print_list = lambda l: print_list(l, self.verbose_print)
    
    @staticmethod
    def n_duplicates(individuals):
        """
        O(n^3) instead of O(n) -> super unefficient!!!
        Problem: individuals not hashable, convienient python syntax
        small lists, so no problem
        """
        counter_lists = ([], [])
        for individual in individuals:
            if individual in counter_lists[0]:
                counter_lists[1][counter_lists[0].index(individual)] += 1
            else:
                counter_lists[0].append(individual)
                counter_lists[1].append(1)

        return sum(counter_lists[1]) - len(counter_lists[0])
    
    def sort_by_fitness(self, individuals_n, generation_naming, add_in_ranking_i_f_n = []):
        """
            input:
            individuals_n = [(individual_1, name_1), (individual_2, name_2), ...]

            output:
            sorted_individuals_f_n = [(individual_1, fitness_1, name_1), (individual_2, fitness_2, name_2), ...]
        """
        progress_bar = lambda progress: self.progress_bar_fitting(generation_naming, progress, len(individuals_n))

        progress = 0
        progress_bar(progress)

        individuals_f_n = []
        for individual, name in individuals_n:
            # rather search through in ineffective cache, than execute the fitness function
            if individual in self.fitness_cache[0]:
                fitness = self.fitness_cache[1][self.fitness_cache[0].index(individual)]
                self.fitness_cache_hits += 1
            else:
                fitness = self.fitness(individual)
                self.fitness_cache[0].append(individual)
                self.fitness_cache[1].append(fitness)
                self.n_calls_fitness += 1
            
            individuals_f_n.append((individual, fitness, name))
            
            progress += 1
            progress_bar(progress)

        # first sort criteria: fitness reverse, second: startswith "old" reverse
        # from small to big, reverse
        individuals_f_n = sorted(individuals_f_n + add_in_ranking_i_f_n, key=lambda x: (x[1], x[2].startswith("old")), reverse=True)

        self.verbose_print('Ranking:')
        self.print_list(individuals_f_n)

        return individuals_f_n
        

    def fit(self, intitial_individual_1, intitial_individual_2, n_generations = 100) -> [dict, float]:
        """
        """
        assert(intitial_individual_1 is not None), "intitial_individual_1 is None"
        assert(intitial_individual_2 is not None), "intitial_individual_2 is None"

        best_individual_f_n, second_best_individual_f_n = self.sort_by_fitness([(intitial_individual_1, "initial 1"), (intitial_individual_2, "initial 2")], "Generation Adam Eva")

        # Generations
        best_individual_f_n_history = [(best_individual_f_n[0], best_individual_f_n[1], "initial")]
        for n_generation in range(n_generations):
            # new generation, evaluate diversity
            individuals_n_new = self.new_generation(best_individual_f_n[0], second_best_individual_f_n[0])
            n_duplicates = self.n_duplicates(list(map(lambda individual_n: individual_n[0], individuals_n_new)))
            self.n_duplicates_in_gens.append(('gen ' + str(n_generation + 1), n_duplicates))
            
            individuals_f_n = self.sort_by_fitness(individuals_n_new, 'Generation ' + str(n_generation + 1) + '/' + str(n_generations), add_in_ranking_i_f_n = [best_individual_f_n, second_best_individual_f_n])

            fitnesses = list(map(lambda individual_n: individual_n[1], individuals_f_n))
            n_duplicates_for_values = list({fitness:fitnesses.count(fitness) for fitness in fitnesses}.values())
            self.n_fitness_duplicates_in_gens.append(('gen ' + str(n_generation + 1), sum(n_duplicates_for_values) - len(n_duplicates_for_values)))

            best_individual_f_n, second_best_individual_f_n = individuals_f_n[:2]
            # if best and second best are the same, then choose the next best for the second best
            if best_individual_f_n[0] == second_best_individual_f_n[0]:
                found_second = False
                for i_f_n in individuals_f_n[2:]:
                    if best_individual_f_n[0] != i_f_n[0]:
                        second_best_individual_f_n = i_f_n
                        found_second = True
                        break
                self.verbose_print("WARNING: all individuals are the same") if not found_second else None

            best_individual_f_n_history.append(best_individual_f_n)

            if not best_individual_f_n[2].startswith("old"):
                self.verbose_print("- new best individual: " + str(best_individual_f_n) + "\n")
            else:
                self.verbose_print("- no better individual\n")
            
            best_individual_f_n = (best_individual_f_n[0], best_individual_f_n[1], "old best")
            second_best_individual_f_n = (second_best_individual_f_n[0], second_best_individual_f_n[1], "old second best")
        
        self.verbose_print("===> EvolutionParamOptimization finished\n")
        avg_n_duplicates = sum(map(lambda x: x[1], self.n_duplicates_in_gens)) / len(self.n_duplicates_in_gens)
        avg_n_fitness_duplicates = sum(map(lambda x: x[1], self.n_fitness_duplicates_in_gens)) / len(self.n_fitness_duplicates_in_gens)
        self.verbose_print("- best individual: " + str(best_individual_f_n) + "\n- n_generations: " + str(n_generations) + "\n- self.n_calls_fitness: " + str(self.n_calls_fitness) + '\n- self.fitness_cache_hits: ' + str(self.fitness_cache_hits) + '\n- self.n_duplicates_in_gens (avg=' + str(avg_n_duplicates) + '): \n\t' + str(self.n_duplicates_in_gens) + '\n- self.n_fitness_duplicates_in_gens (avg=' + str(avg_n_fitness_duplicates) + '): \n\t' + str(self.n_fitness_duplicates_in_gens) + '\n')
        self.verbose_print("- best individual history:\n")
        self.print_list(best_individual_f_n_history)
        self.verbose_print("\n\n=======================================\n")

        return best_individual_f_n, best_individual_f_n_history
    
    @staticmethod
    def test_fit():
        """
        Test Evolution Optimizer
        dummy fitness function with hardcoded optimum
        - to see, how close the evolution optimization will come to this optimum
        - fitness function executes fast, to see the whole evolution process faster
        """
        gen_pool = [
            ("max_depth", "natural", [0, 20]),
            ("n_estimators", "natural", [10, 200]),
            ("learning_rate", "real", [0.01, 0.4]),
            ("tree_method", "categorical", ["gpu_hist", "gpu_hist_bins", "gpu_approx", "hist", "approx"])
        ]

        xgb_dna = XGBHyperparamDNA(gen_pool)

        def log_func(*args, **kwargs):
            print(*args, **kwargs)
            # if len(args) > 0 and args[0] != '\n':
            #     msg = ''.join(list(map(str, args)))
            #     send_telegram(msg)


        def fitness(individual):
            fitness = -20 # sigmoid between -6 and 6

            # learning_rate
            optimal_learning_rate = 0.1253875
            distance = abs(optimal_learning_rate - individual['learning_rate'])
            fitness += ((1 - distance)**5)*8 # can come to +6 -> 0 -> simoid 50 percent
            
            # n_estimators
            optimal_n_estimators = 100
            max_distance = 100
            distance = abs(optimal_n_estimators - individual['n_estimators']) # between 0 and 100
            fitness += 6 * (distance / max_distance) # can get max 3 improvement

            # max_depth
            optimal_max_depth = 5
            max_distance = 15
            distance = abs(optimal_max_depth - individual['max_depth']) # between 0 and 15
            fitness += 6 * (distance / max_distance) # can get max 3 improvement

            # tree_method
            if individual['tree_method'] == 'gpu_hist':
                fitness += 2
            elif individual['tree_method'] == 'gpu_hist_bins':
                fitness += 1.5
            elif individual['tree_method'] == 'gpu_approx':
                fitness += 1
            elif individual['tree_method'] == 'hist':
                fitness += 0.5
            elif individual['tree_method'] == 'approx':
                fitness += 0.25

            time.sleep(0.1)
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            return sigmoid(fitness)

        evo_optimizer = EvolutionOptimizer(xgb_dna.new_generation, fitness, log_func=log_func)

        initial_individuals = [
            {'max_depth': 10, 'n_estimators': 10, 'learning_rate': 0.4, 'tree_method': 'approx'},
            {'max_depth': 0, 'n_estimators': 10, 'learning_rate': 0.4, 'tree_method': 'gpu_hist_bins'}
        ]

        best_individual_f_n, best_individual_f_n_history = evo_optimizer.fit(initial_individuals[0], initial_individuals[1], n_generations=10)
        return best_individual_f_n, best_individual_f_n_history
            
