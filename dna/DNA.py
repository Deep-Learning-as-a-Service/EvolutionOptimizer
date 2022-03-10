import random
from utils.utils import print_list, dicts_equal_n_exceptions
import warnings


class DNA():

    def __init__(self, gen_pool):
        """
            example:
            
            gen_pool = [
                ("max_depth", "natural", [0, 20]),
                ("n_estimators", "natural", [10, 200]),
                ("learning_rate", "real", [0.01, 0.4]),
                ("tree_method", "categorical", ["gpu_hist", "gpu_hist_bins", "gpu_approx", "hist", "approx"])
            ]
        """
        self.gen_pool = gen_pool

    def new_generation(self, individual_1, individual_2) -> list:
        """
        better the less duplicates appear over the calls and within the generations

        returns
            [(individual_1, name_1), (individual_2, name_2), ...]
        """

        raise NotImplementedError("new_generation")
    
    def test_new_generation(self, best_individual, second_best_individual):
        print("test_new_generation start")
        for i in range(3):
            print("generation ", i + 1, ":")
            print("best_individual: ", best_individual)
            print("second_best_individual: ", second_best_individual)
            print_list(self.new_generation(best_individual, second_best_individual))
            print()
        print("test_new_generation end")

