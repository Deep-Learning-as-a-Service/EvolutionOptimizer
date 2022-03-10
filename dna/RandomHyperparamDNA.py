from dna.HyperparamDNA import HyperparamDNA

class RandomHyperparamDNA(HyperparamDNA):
    def new_generation(self, best_individual, second_best_individual):
        return [(self._random_individual(), 'rand ' + str(i + 1)) for i in range(8)]