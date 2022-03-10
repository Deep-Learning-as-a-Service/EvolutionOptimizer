from dna.DNA import DNA
import random

class SimpleDeepNetDNA(DNA):
    """
        example:

        gen_pool = {
            'n_layers_range': [1, 4], # net meta var

            'units': [0, 32], # n_neurons_per_layer_range

            # later to add
            'activation_function_range': ['relu', 'tanh', 'sigmoid', 'softmax'],
            'optimizer_range': ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam'],
            'loss_function_range': ['categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity', 'is_categorical_crossentropy', 'is_mean_squared_error', 'is_mean_absolute_error', 'is_mean_absolute_percentage_error', 'is_mean_squared_logarithmic_error', 'is_squared_hinge', 'is_hinge', 'is_categorical_hinge', 'is_logcosh', 'is_categorical_crossentropy', 'is_sparse_categorical_crossentropy', 'is_kullback_leibler_divergence', 'is_poisson', 'is_cosine_proximity'],
            'batch_size_range': [1, 64],
            'epochs_range': [1, 100],
            'verbose_range': [0, 2],
            'validation_split_range': [0.0, 0.2],
            'early_stopping_range': [0, 10],
            'shuffle_range': [0, 1],
            'dropout_range': [0.0, 0.5],
            'weight_constraint_range': [0, 10],
            'kernel_initializer_range': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
            'bias_initializer_range': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
            'kernel_regularizer_range': ['l1', 'l2', 'l1_l2'],
            'bias_regularizer_range': ['l1', 'l2', 'l1_l2'],
            'activity_regularizer_range': ['l1', 'l2', 'l1_l2'],
            'kernel_constraint_range': ['max_norm', 'non_neg'],
            'bias_constraint_range': ['max_norm', 'non_neg'],
        }

        individual = [8, 12] # net_config: 8-8-12-1
    """
    def __init__(self, gen_pool):
        super().__init__(gen_pool)
        self.__mutation_rate = 0.5
        self.__max_mutation_intensity = 0.2
    
    @staticmethod
    def test():
        gen_pool = {
            'n_layers_range': [1, 4], # net meta var
            'units': [0, 32], # n_neurons_per_layer_range
        }

        dna = SimpleDeepNetDNA(gen_pool)
        dna.test_new_generation(
            [8, 12, 32, 5],
            [5, 5],
        )

    def new_generation(self, best_individual, second_best_individual) -> list:
        return [
            (self._cross_individual(best_individual, second_best_individual), 'cross 1'),
            (self._cross_individual(best_individual, second_best_individual), 'cross 2'),
            (self.__cross_between_individual(best_individual, second_best_individual), 'cross between 1'),
            (self.__cross_between_individual(best_individual, second_best_individual), 'cross between 2'),
            (self._mutate_individual(best_individual), 'mutate best'),
            (self._mutate_individual(second_best_individual), 'mutate second best'),
            (self._random_individual(), 'rand 1'),
            (self._random_individual(), 'rand 2')
        ]
    
    def __random_int_range(self, range):
        assert len(range) == 2, "Range must be a list of two integers"

        if range[0] <= range[1]:
            return random.randint(range[0], range[1])
        else:
            return random.randint(range[1], range[0])


    def _random_individual(self):
        n_layers = self.__random_int_range(self.gen_pool['n_layers_range'])
        return [random.randint(0, 32) for _ in range(n_layers)]
    
    def get_random_individuals(self, n_individuals):
        return [self._random_individual() for _ in range(n_individuals)]
    
    def __duplicate_protection(self, creation_func, individuals, max_tries=100):
        max_tries_initial = max_tries
        new_individual = creation_func()
        while new_individual in individuals and max_tries > 0:
            new_individual = creation_func()
            max_tries -= 1
        if max_tries == 0:
            raise Exception("Max individual creation attempts (" + str(max_tries_initial) + ") reached")
            # print("WARNING: Max individual creation attempts (" + str(max_tries_initial) + ") reached")
            # return self._random_individual()
        return new_individual
    
    def __mutate_until(self, should_criteria, individual_1, individual_2):
        
        new_mutation = lambda: self._mutate_individual(individual_2)
        criteria = lambda: should_criteria(individual_1, individual_2)
        if criteria(): return individual_1, individual_2
        individual_2 = new_mutation()

        max_tries_initial = 100
        max_tries = max_tries_initial
        while not criteria() and max_tries > 0:
            individual_2 = new_mutation()
            max_tries -= 1
        if max_tries == 0: raise Exception("__mutate_until - Max individual creation attempts (" + str(max_tries_initial) + ") reached") # warning go random before you do that!
        
        return individual_1, individual_2

    
    def __cross_between_individual(self, individual1, individual2):
        individual1, individual2 = self.__mutate_until(lambda i1, i2: i1 != i2, individual1, individual2)
        def creation_func():
            n_layers = self.__random_int_range([len(individual1), len(individual2)])
            units_range = [min(min(individual1), min(individual2)), max(max(individual1), max(individual2))]
            return [self.__random_int_range(units_range) for _ in range(n_layers)]
        return self.__duplicate_protection(creation_func, [individual1, individual2])
    
    def _cross_individual(self, individual1, individual2):
        individual1, individual2 = self.__mutate_until(lambda i1, i2: i1 != i2, individual1, individual2)
        def creation_func():
            n_layers = self.__random_int_range([len(individual1), len(individual2)])
            return [random.choice(individual1 + individual2) for _ in range(n_layers)]
        return self.__duplicate_protection(creation_func, [individual1, individual2])
    
    def _mutate_individual(self, individual):
        def creation_func():
            max_change_value = int(self.__max_mutation_intensity * (self.gen_pool['units'][1] - self.gen_pool['units'][0]))
            new_individual = []
            for i in range(len(individual)):
                new_units = individual[i]
                if random.random() < self.__mutation_rate: 
                    new_units_suggestion = new_units + random.randint(-max_change_value, max_change_value)
                    # only change if new_units_suggestion is in range
                    if new_units_suggestion >= self.gen_pool['units'][0] and new_units_suggestion <= self.gen_pool['units'][1]:
                        new_units = new_units_suggestion
                new_individual.append(new_units)
            return new_individual

        return self.__duplicate_protection(creation_func, [individual])