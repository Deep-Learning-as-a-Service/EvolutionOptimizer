import json
from evolution.EvolutionOptimizer import EvolutionOptimizer

def origin_improvement(history):
    origin_improvement = []
    """
    for every history there is a
    origin_improvement = [
        ('gen 4', 'rand 2', 0.15579427753781183),
        ('gen 7', 'mutate best', 0.09609640185207269)
    ]
    """

    last_best_fitness = 0
    for i, h_entry in enumerate(history):
        individual, fitness, origin = h_entry
        if (origin != 'old best') and (origin != 'initial best'):
            improvement = fitness - last_best_fitness
            origin_improvement.append(('gen ' + str(i), origin, improvement))
        last_best_fitness = fitness
    
    return origin_improvement



# do all evos ---------------------------------------------------------------
n_evos = 3

best_individuals_fitness_evo = []
"""
best_individuals_fitness_evo = [
    ('evo 1', {'max_depth': 10, 'n_estimators': 10, 'learning_rate': 0.4, 'tree_method': 'approx'}, 0.723),
    ('evo 2', {'max_depth': 10, 'n_estimators': 20, 'learning_rate': 0.4, 'tree_method': 'approx'}, 0.423),
    ...
]
"""
origin_improvements = []
"""
    origin_improvements = [
        ('evo 0', [
                    ('gen 4', 'rand 2', 0.15579427753781183),
                    ('gen 7', 'mutate best', 0.09609640185207269)
                ]),
        ('evo 1', [
                    ('gen 4', 'rand 2', 0.15579427753781183),
                    ('gen 7', 'mutate best', 0.09609640185207269)
                ])
        ...
    ]
"""

for i in range(n_evos):
    best_individual_f_n, best_individual_f_n_history = EvolutionOptimizer.test_fit()
    origin_improvements.append(('evo ' + str(i + 1), origin_improvement(best_individual_f_n_history)))

    best_individual, best_individual_fitness, _ = best_individual_f_n
    best_individuals_fitness_evo.append(('evo ' + str(i), best_individual, best_individual_fitness))

# --------------------------------------------------------------------------


origin_improvements_evaluation = {}
"""
    origin_improvements_evaluation = {

        'rand 2': (1.235, ['gen 4', 'gen 2', 'gen 1']), 
        ...
"""
for origin_improvement in origin_improvements:
    for origin_entry in origin_improvement[1]:
        gen, origin, improvement = origin_entry
        if origin not in origin_improvements_evaluation:
            origin_improvements_evaluation[origin] = (improvement, [gen])
        else:
            improvement_so_far, gen_list = origin_improvements_evaluation[origin]
            origin_improvements_evaluation[origin] = (improvement_so_far + improvement, gen_list + [gen])

origin_improvements_evaluation = dict(sorted(origin_improvements_evaluation.items(), key=lambda dict_tuple: dict_tuple[1][0], reverse=True))

print('====> Meta 10 Evos Evaluation:\n')
"""
    Expectatation:
    - random should deliver the best individual in few cases, but big improvements
    - mutation and cross over should deliver most of the best individual improvements, but small ones
"""

print('==> origin_improvements_evaluation:\n')
print(json.dumps(origin_improvements_evaluation, indent=2))
print()

print("==> origin_improvements for each of the " + str(n_evos) + " evos:\n")
print(json.dumps(origin_improvements, indent=2))
# print_list(origin_improvements)
print()

print("==> best_individuals_fitness_evo for each of the " + str(n_evos) + " evos:\n")
print(json.dumps(best_individuals_fitness_evo, indent=2))
# print_list(best_individuals_fitness_evo)
print()
