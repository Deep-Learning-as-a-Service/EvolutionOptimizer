from dna.DNA import DNA
from utils.utils import print_list, dicts_equal_n_exceptions
import random


class XGBHyperparamDNA(DNA):

    def __init__(self, gen_pool, mutation_rate=0.5):
        super().__init__(gen_pool)
        self.mutation_rate = mutation_rate
        self.slightly_mutation_range = 0.1

    @staticmethod
    def test():
        xgb_gen_pool = [
            ("max_depth", "natural", [3, 20]),
            ("n_estimators", "natural", [100, 1000]),
            ("learning_rate", "real", [0.01, 0.4]),
            ("tree_method", "categorical", ["gpu_hist", "gpu_hist_bins", "gpu_approx", "hist", "approx"])
        ]

        xgb_dna = XGBHyperparamDNA(xgb_gen_pool)
        xgb_dna.test_new_generation(
            { "max_depth": 15, "n_estimators": 300, "learning_rate": 0.3, "tree_method": "gpu_hist"},
            { "max_depth": 5, "n_estimators": 100, "learning_rate": 0.1, "tree_method": "gpu_hist_bins" }
        )

    def new_generation(self, best_individual, second_best_individual) -> list:
        return [
            (self.__cross_individual(best_individual, second_best_individual), 'cross 1'),
            (self.__cross_individual(best_individual, second_best_individual), 'cross 2'),
            (self.__mutate_individual(best_individual), 'mutate best'),
            (self.__mutate_individual(second_best_individual), 'mutate second best'),
            (self.__mutate_slightly_individual(best_individual), 'mutate slightly best'),
            (self.__mutate_slightly_individual(second_best_individual), 'mutate slightly second best'),
            (self.__random_individual(), 'rand 1'),
            (self.__random_individual(), 'rand 2')
        ]
    
    def two_random_individuals(self):
        return self.__random_individual(), self.__random_individual(), 'eva rand'
    
    def __random_gene_value(_, gene):
        name, type, range = gene

        if type == "natural":
            return random.randint(range[0], range[1])
        elif type == "real":
            return round(random.uniform(range[0], range[1]), ndigits=6)
        elif type == "categorical":
            return random.choice(range)
        else:
            raise Exception("Unknown gene type: " + type)

    def __individual_from_func(self, new_gene_value):
        """
            new_gene_value: (gene) -> new_gene_value
            gene: (name, type, range)
            - its important to create a new individual here #deepcopy
        """
        return {gene[0]: new_gene_value(gene) for gene in self.gen_pool}
    
    def __duplicate_protection(self, creation_func, individuals, max_tries=100):
        max_tries_initial = max_tries
        new_individual = creation_func()
        while new_individual in individuals and max_tries > 0:
            new_individual = creation_func()
            max_tries -= 1
        if max_tries == 0:
            raise Exception("Max individual creation attempts (" + str(max_tries_initial) + ") reached")
            # print("WARNING: Max individual creation attempts (" + str(max_tries_initial) + ") reached")
            # return self.__random_individual()
        return new_individual
    
    def __mutate_slightly_until(self, should_criteria, individual_1, individual_2):
        
        new_mutation = lambda: self.__mutate_slightly_individual(individual_2)
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

    
    def __random_individual(self):
        return self.__individual_from_func(self.__random_gene_value)
    
    def __cross_individual(self, individual_1, individual_2):
        """
        if you cross individuals, that will be in the new generation, this could generate duplicates, so here a protection against that
        - consider a random value in parents range for natural and real values
        """
        # input individuals, need to be "different enough" to create useful crossing, that add value to the diversity
        diversity_criteria = lambda individual_1, individual_2: not dicts_equal_n_exceptions(individual_1, individual_2, n_exceptions = 1)
        if not diversity_criteria(individual_1, individual_2): 
            individual_1, individual_2 = self.__mutate_slightly_until(diversity_criteria, individual_1, individual_2)
        
        creation_func = lambda: self.__individual_from_func(lambda gene: individual_1[gene[0]] if random.random() > 0.5 else individual_2[gene[0]])
        return self.__duplicate_protection(creation_func, [individual_1, individual_2])

    def __mutate_individual(self, individual):
        creation_func = lambda: self.__individual_from_func(lambda gene: self.__random_gene_value(gene) if random.random() < self.mutation_rate else individual[gene[0]])
        return self.__duplicate_protection(creation_func, [individual])

    def __mutate_slightly_individual(self, individual):
        def new_gene_value(gene):
            name, type, range = gene
            if random.random() < self.mutation_rate:
                current_value = individual[name]

                if type == "natural":
                    mutation_range = int((range[1] - range[0]) * self.slightly_mutation_range)
                    new_value = random.randint(current_value - mutation_range, current_value + mutation_range)
                    if new_value < range[0]: new_value = range[0]
                    elif new_value > range[1]: new_value = range[1]

                elif type == "real":
                    mutation_range = (range[1] - range[0]) * self.slightly_mutation_range
                    new_value = round(random.uniform(current_value - mutation_range, current_value + mutation_range), ndigits=6)
                    if new_value < range[0]: new_value = range[0]
                    elif new_value > range[1]: new_value = range[1]

                elif type == "categorical":
                    new_value = random.choice(range)
                else:
                    raise Exception("Unknown gene type: " + type)

                return new_value
            else:
                return individual[name]
                
        creation_func = lambda: self.__individual_from_func(new_gene_value)
        return self.__duplicate_protection(creation_func, [individual])