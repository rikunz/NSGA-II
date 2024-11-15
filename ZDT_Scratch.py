import numpy as np
from copy import deepcopy
from itertools import chain
import matplotlib.pyplot as plt

class NSGA2:
    def __init__(self, 
                 max_iter=100,
                 pop_size=100,
                 p_crossover=0.8,
                 p_mutation=0.2,
                 eta_cross=20.0,
                 eta_mutate=20.0,
                 verbose=True):
        """Constructor for the NSGA-II object with improved parameters"""
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.eta_cross = eta_cross  # Distribution parameter for SBX crossover
        self.eta_mutate = eta_mutate  # Distribution parameter for polynomial mutation
        self.verbose = verbose

    def run(self, problem):
        """Runs the NSGA-II algorithm on a given problem."""
        
        # Extract Problem Info
        cost_function = problem['cost_function']
        n_var = problem['n_var']
        var_size = (n_var,)
        
        # Separate bounds for x1 and xi
        x1_bounds = problem['x1_bounds']
        xi_bounds = problem['xi_bounds']
        
        # Create bounds array for all variables
        var_min = np.array([x1_bounds[0]] + [xi_bounds[0]] * (n_var - 1))
        var_max = np.array([x1_bounds[1]] + [xi_bounds[1]] * (n_var - 1))

        # Number of offsprings (multiple of 2)
        n_crossover = 2 * int(self.p_crossover * self.pop_size / 2)
        
        # Number of Mutants
        n_mutation = int(self.p_mutation * self.pop_size)

        # Empty Individual Template
        empty_individual = {
            'position': None,
            'cost': None,
            'rank': None,
            'crowding_distance': None,
        }

        # Initialize Population
        pop = [deepcopy(empty_individual) for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            # Initialize x1 and xi separately
            pop[i]['position'] = np.zeros(n_var)
            pop[i]['position'][0] = np.random.uniform(x1_bounds[0], x1_bounds[1])
            pop[i]['position'][1:] = np.random.uniform(xi_bounds[0], xi_bounds[1], n_var - 1)
            pop[i]['cost'] = cost_function(pop[i]['position'])

        # Non-dominated Sorting
        pop, F = self.non_dominated_sorting(pop)

        # Calculate Crowding Distance
        pop = self.calc_crowding_distance(pop, F)

        # Sort Population
        pop, F = self.sort_population(pop)

        # Main Loop
        for it in range(self.max_iter):
            
            # Crossover
            popc = [[deepcopy(empty_individual), deepcopy(empty_individual)] for _ in range(n_crossover//2)]
            for k in range(n_crossover//2):
                # Tournament Selection
                parents = self.tournament_selection(pop, 2)
                p1, p2 = pop[parents[0]], pop[parents[1]]
                
                # Simulated Binary Crossover (SBX)
                popc[k][0]['position'], popc[k][1]['position'] = self.sbx_crossover(
                    p1['position'], p2['position'], var_min, var_max)
                
                popc[k][0]['cost'] = cost_function(popc[k][0]['position'])
                popc[k][1]['cost'] = cost_function(popc[k][1]['position'])
            
            # Flatten Offsprings List
            popc = list(chain(*popc))
            
            # Mutation
            popm = [deepcopy(empty_individual) for _ in range(n_mutation)]
            for k in range(n_mutation):
                p = pop[np.random.randint(self.pop_size)]
                popm[k]['position'] = self.polynomial_mutation(
                    p['position'].copy(), var_min, var_max)
                popm[k]['position'] = np.clip(popm[k]['position'], var_min, var_max)
                popm[k]['cost'] = cost_function(popm[k]['position'])

            # Create Merged Population
            pop = pop + popc + popm

            # Non-dominated Sorting
            pop, F = self.non_dominated_sorting(pop)

            # Calculate Crowding Distance
            pop = self.calc_crowding_distance(pop, F)
            
            # Sort Population
            pop, F = self.sort_population(pop)

            # Truncate Population
            pop, F = self.truncate_population(pop, F)

            if self.verbose and (it + 1) % 10 == 0:
                print(f'Iteration {it + 1}: Number of Pareto Members = {len(F[0])}')

        pareto_pop = [pop[i] for i in F[0]]
        population_dominated = [p for i, p in enumerate(pop) if i not in F[0]]
        
        return {
            'pop': pop,
            'F': F,
            'pareto_pop': pareto_pop,
            'dominated': population_dominated
        }

    def tournament_selection(self, pop, tournament_size):
        """Tournament selection for parent selection"""
        indices = np.random.choice(len(pop), size=tournament_size, replace=False)
        tournament = [pop[i] for i in indices]
        return sorted(indices, key=lambda i: (pop[i]['rank'], -pop[i]['crowding_distance']))

    def sbx_crossover(self, x1, x2, var_min, var_max):
        """Simulated Binary Crossover (SBX)"""
        y1 = x1.copy()
        y2 = x2.copy()
        
        if np.random.random() <= self.p_crossover:
            for i in range(len(x1)):
                if np.random.random() <= 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if x1[i] < x2[i]:
                            parent1, parent2 = x1[i], x2[i]
                        else:
                            parent1, parent2 = x2[i], x1[i]
                            
                        beta = 1.0 + (2.0 * (parent1 - var_min[i]) / (parent2 - parent1))
                        alpha = 2.0 - beta ** (-(self.eta_cross + 1.0))
                        
                        rand = np.random.random()
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (self.eta_cross + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_cross + 1.0))
                        
                        child1 = 0.5 * ((parent1 + parent2) - beta_q * (parent2 - parent1))
                        child2 = 0.5 * ((parent1 + parent2) + beta_q * (parent2 - parent1))
                        
                        child1 = np.clip(child1, var_min[i], var_max[i])
                        child2 = np.clip(child2, var_min[i], var_max[i])
                        
                        if np.random.random() <= 0.5:
                            y1[i] = child2
                            y2[i] = child1
                        else:
                            y1[i] = child1
                            y2[i] = child2
                            
        return y1, y2

    def polynomial_mutation(self, x, var_min, var_max):
        """Polynomial Mutation"""
        y = x.copy()
        
        for i in range(len(x)):
            if np.random.random() <= self.p_mutation:
                delta_1 = (y[i] - var_min[i]) / (var_max[i] - var_min[i])
                delta_2 = (var_max[i] - y[i]) / (var_max[i] - var_min[i])
                
                rand = np.random.random()
                mut_pow = 1.0 / (self.eta_mutate + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_mutate + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_mutate + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                y[i] = y[i] + delta_q * (var_max[i] - var_min[i])
                y[i] = np.clip(y[i], var_min[i], var_max[i])
                
        return y

    def dominates(self, p, q):
        """Checks if p dominates q"""
        return all(p['cost'] <= q['cost']) and any(p['cost'] < q['cost'])

    def non_dominated_sorting(self, pop):
        """Perform Non-dominated Sorting on a Population"""
        pop_size = len(pop)

        # Initialize Domination Stats
        domination_set = [[] for _ in range(pop_size)]
        dominated_count = [0 for _ in range(pop_size)]

        # Initialize Pareto Fronts
        F = [[]]  # First front

        # Find the first Pareto Front
        for i in range(pop_size):
            for j in range(i+1, pop_size):
                if self.dominates(pop[i], pop[j]):
                    domination_set[i].append(j)
                    dominated_count[j] += 1
                elif self.dominates(pop[j], pop[i]):
                    domination_set[j].append(i)
                    dominated_count[i] += 1

            if dominated_count[i] == 0:
                pop[i]['rank'] = 0
                F[0].append(i)

      
        i = 0
        while i < len(F):  
            Q = []  
            for p in F[i]:
                for q in domination_set[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        pop[q]['rank'] = i + 1
                        Q.append(q)
            i += 1
            if Q:  # Only append non-empty fronts
                F.append(Q)

        return pop, F

    def calc_crowding_distance(self, pop, F):
        """Calculate the crowding distance for a given population"""
        for front in F:
            if len(front) > 0:
                solutions_num = len(front)
                obj_num = len(pop[0]['cost'])
                
                for individual in front:
                    pop[individual]['crowding_distance'] = 0

                for m in range(obj_num):
                    front = sorted(front, key=lambda x: pop[x]['cost'][m])
                    
                    pop[front[0]]['crowding_distance'] = float('inf')
                    pop[front[-1]]['crowding_distance'] = float('inf')
                    
                    obj_max = pop[front[-1]]['cost'][m]
                    obj_min = pop[front[0]]['cost'][m]
                    
                    if obj_max - obj_min > 0:
                        for i in range(1, solutions_num-1):
                            pop[front[i]]['crowding_distance'] += \
                                (pop[front[i+1]]['cost'][m] - pop[front[i-1]]['cost'][m]) / \
                                (obj_max - obj_min)

        return pop
    
    def sort_population(self, pop):
        """Sorts population based on rank and crowding distance"""
        pop = sorted(pop, key=lambda x: (x['rank'], -x['crowding_distance']))
        
        max_rank = pop[-1]['rank']
        F = [[] for _ in range(max_rank + 1)]
        for i, individual in enumerate(pop):
            F[individual['rank']].append(i)
            
        return pop, F
    
    def truncate_population(self, pop, F, pop_size=None):
        """Truncates population to specified size"""
        if pop_size is None:
            pop_size = self.pop_size

        if len(pop) <= pop_size:
            return pop, F

        pop = pop[:pop_size]
        F = [[i for i in front if i < pop_size] for front in F]
        F = [front for front in F if front] 
        
        return pop, F



class Problem:
    def evalZDT2(individual):
        x1 = individual[0]
        n = len(individual)
        g = 1 + (9 / (n - 1)) * sum(x**2 for x in individual[1:])
        f1 = x1
        f2 = g * (1 - (x1 / g) ** 2)
        return np.array([f1, f2])


def plot_pareto_front(pareto_solutions, dominated, show_optimal=True, figsize=(10, 6)):
    """
    Memvisualisasikan solusi pareto front.
    """

    f1_values = [f['cost'][0] for f in pareto_solutions]
    f2_values = [f['cost'][1] for f in pareto_solutions]
    
    plt.figure(figsize=figsize)
    plt.scatter(f1_values, f2_values, color='blue', label="Pareto Solutions")

    if dominated:
        f1_dom = [ind['cost'][0] for ind in dominated]
        f2_dom = [ind['cost'][1] for ind in dominated]
        plt.scatter(f1_dom, f2_dom, c='r', marker='x', s=30, label='Dominated Individu')
        
    if show_optimal:
        x1_values = np.linspace(0, 1, 100)
        f2_values = 1 - x1_values ** 2 
        plt.plot(x1_values, f2_values, color='red', label="Pareto Optimal Front")
    
    plt.xlabel("Objective 1 (f1)")
    plt.ylabel("Objective 2 (f2)")
    plt.legend()
    plt.title("Scatter Plot of Pareto Solutions with Pareto Optimal Front")
    plt.grid(True)
    plt.show()


problem = {
    'cost_function': Problem.evalZDT2,
    'n_var': 30,
    'x1_bounds': (0, 1),    # Bounds for x1
    'xi_bounds': (-1, 1),   # Bounds for other variables
}


nsga2 = NSGA2(
    max_iter=100,
    pop_size=150,
    p_crossover=0.8,
    p_mutation=0.2,
    eta_cross=20.0,
    eta_mutate=20.0,
    verbose=True
)

results = nsga2.run(problem)

pareto = results['pareto_pop']
dominated = results['dominated']
print(pareto)
plot_pareto_front(pareto, dominated)