import numpy as np
from copy import deepcopy
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NSGA2:
    def __init__(self, 
                 max_iter=250,
                 pop_size=100,
                 p_crossover=0.9,
                 p_mutation=0.1,
                 eta_cross=20.0,
                 eta_mutate=20.0,
                 verbose=True):
        """Constructor for the NSGA-II object with parameters matching DEAP implementation"""
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.eta_cross = eta_cross
        self.eta_mutate = eta_mutate
        self.verbose = verbose

    def run(self, problem):
        """Runs the NSGA-II algorithm for the comet problem."""
        
        # Extract Problem Info
        cost_function = problem['cost_function']
        n_var = 3  # Fixed for comet problem
        var_size = (n_var,)
        
        # Bounds for each variable
        x1_bounds = problem['x1_bounds']
        x2_bounds = problem['x2_bounds']
        x3_bounds = problem['x3_bounds']
        
        # Create bounds arrays
        var_min = np.array([x1_bounds[0], x2_bounds[0], x3_bounds[0]])
        var_max = np.array([x1_bounds[1], x2_bounds[1], x3_bounds[1]])

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
            pop[i]['position'] = np.array([
                np.random.uniform(x1_bounds[0], x1_bounds[1]),
                np.random.uniform(x2_bounds[0], x2_bounds[1]),
                np.random.uniform(x3_bounds[0], x3_bounds[1])
            ])
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
            popc = []
            for _ in range(n_crossover//2):
                # Tournament Selection
                parents = self.tournament_selection(pop, 2)
                p1, p2 = pop[parents[0]], pop[parents[1]]
                
                # Create offspring using SBX
                c1, c2 = deepcopy(empty_individual), deepcopy(empty_individual)
                c1['position'], c2['position'] = self.sbx_crossover(
                    p1['position'], p2['position'], var_min, var_max)
                
                c1['cost'] = cost_function(c1['position'])
                c2['cost'] = cost_function(c2['position'])
                
                popc.extend([c1, c2])
            
            # Mutation
            popm = []
            for _ in range(n_mutation):
                # Select parent
                p = pop[np.random.randint(self.pop_size)]
                
                # Create mutant
                m = deepcopy(empty_individual)
                m['position'] = self.polynomial_mutation(
                    p['position'].copy(), var_min, var_max)
                m['cost'] = cost_function(m['position'])
                popm.append(m)

            # Merge Population
            pop = pop + popc + popm

            # Non-dominated Sorting
            pop, F = self.non_dominated_sorting(pop)

            # Calculate Crowding Distance
            pop = self.calc_crowding_distance(pop, F)
            
            # Sort Population
            pop, F = self.sort_population(pop)

            # Truncate Population
            pop, F = self.truncate_population(pop, F)

            if self.verbose and (it + 1) % 50 == 0:
                print(f'Iteration {it + 1}: Number of Pareto Members = {len(F[0])}')

        pareto_front = [pop[i] for i in F[0]]
        population_dominated = [p for i, p in enumerate(pop) if i not in F[0]]
        
        return pareto_front, population_dominated

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
        domination_set = [[] for _ in range(pop_size)]
        dominated_count = [0 for _ in range(pop_size)]
        F = [[]]

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
            if Q:
                F.append(Q)

        return pop, F

    def calc_crowding_distance(self, pop, F):
        """Calculate crowding distance for each solution"""
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
                    
                    obj_range = pop[front[-1]]['cost'][m] - pop[front[0]]['cost'][m]
                    if obj_range > 0:
                        for i in range(1, solutions_num-1):
                            pop[front[i]]['crowding_distance'] += \
                                (pop[front[i+1]]['cost'][m] - pop[front[i-1]]['cost'][m]) / obj_range

        return pop
    
    def sort_population(self, pop):
        """Sort population based on rank and crowding distance"""
        pop = sorted(pop, key=lambda x: (x['rank'], -x['crowding_distance']))
        
        max_rank = pop[-1]['rank']
        F = [[] for _ in range(max_rank + 1)]
        for i, individual in enumerate(pop):
            F[individual['rank']].append(i)
            
        return pop, F
    
    def truncate_population(self, pop, F, pop_size=None):
        """Truncate population to specified size"""
        if pop_size is None:
            pop_size = self.pop_size

        if len(pop) <= pop_size:
            return pop, F

        pop = pop[:pop_size]
        F = [[i for i in front if i < pop_size] for front in F]
        F = [front for front in F if front]
        
        return pop, F

def plot_solutions(pareto_front, population_dominated=None):
    """Plot the Pareto front and dominated solutions"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    

    f1_values = [ind['cost'][0] for ind in pareto_front]
    f2_values = [ind['cost'][1] for ind in pareto_front]
    f3_values = [ind['cost'][2] for ind in pareto_front]
    ax.scatter(f1_values, f2_values, f3_values, c='b', marker='o', s=50, label='Pareto Front')
    

    if population_dominated:
        f1_dom = [ind['cost'][0] for ind in population_dominated]
        f2_dom = [ind['cost'][1] for ind in population_dominated]
        f3_dom = [ind['cost'][2] for ind in population_dominated]
        ax.scatter(f1_dom, f2_dom, f3_dom, c='r', marker='x', s=30, label='Dominated Solutions')
    
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.legend()
    plt.title('3D Pareto Front for Comet Problem')
    plt.show()


class CometProblem:
    @staticmethod
    def evaluate(x):
        x1, x2, x3 = x
        f1 = (1 + x3) * (x1**3 * x2**2 - 10 * x1 - 4 * x2)
        f2 = (1 + x3) * (x1**3 * x2**2 - 10 * x1 + 4 * x2)
        f3 = 3 * (1 + x3) * x1**2
        return np.array([f1, f2, f3])


problem = {
    'cost_function': CometProblem.evaluate,
    'x1_bounds': (1, 3.5),
    'x2_bounds': (-2, 2),
    'x3_bounds': (0, 1)
}

nsga2 = NSGA2(
    max_iter=100,
    pop_size=350,
    p_crossover=0.9,
    p_mutation=0.1,
    eta_cross=20.0,
    eta_mutate=20.0,
    verbose=True
)

pareto_front, dominated = nsga2.run(problem)
plot_solutions(pareto_front, dominated)