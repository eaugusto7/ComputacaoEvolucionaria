import random
import numpy as np
import math

class Bee:
    def __init__(self, position):
        self.position = position
        self.fitness = evaluate_fitness(position)

def langermann(xx):
    m = 5
    c = [1, 2, 5, 2, 3]
    A = [[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]

    outer = 0.0
    for i in range(m):
        inner = 0.0
        inner += (xx[0] - A[i][0]) ** 2
        inner += (xx[1] - A[i][1]) ** 2
        newTerm = c[i] * math.exp(-inner / math.pi) * math.cos(math.pi * inner)
        outer += newTerm
    return outer

def deJong5(xx):
    a1 = [-32.0, -16.0, 0.0, 16.0, 32.0]*5
    a2 = [-32.0]*5 + [-16.0]*5 + [0.0]*5 + [16.0]*5 + [32.0]*5
    a = [a1, a2]

    sum = 0.002
    for i in range(1, 26):
        sum += ( 1 / (i + (xx[0] - a[0][i-1])**6 + (xx[1] - a[1][i-1])**6) )
    
    return 1.0/sum

def schwef(xx):
    d = len(xx)
    _sum = 0
    for xi in xx:
        _sum += xi * np.sin(np.sqrt(abs(xi)))
    
    y = 418.9829 * d - _sum
    return y

def levy(xx):
    d = len(xx)
    
    w = np.zeros(d)
    for ii in range(d):
        w[ii] = 1 + (xx[ii] - 1) / 4
    
    term1 = (np.sin(np.pi * w[0])) ** 2
    term3 = (w[d - 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[d - 1])) ** 2)
    
    _sum = 0
    for ii in range(d - 1):
        wi = w[ii]
        new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
        _sum += new
    
    y = term1 + _sum + term3
    
    return y

def evaluate_fitness(position):
    #return langermann(position)
    return deJong5(position)
    #return schwef(position)
    #return levy(position)

def employedBees(population, bottom_limit, higher_limit):
    for bee in population:
        new_position = bee.position + np.random.uniform(low=-1, high=1, size=len(bee.position))
        new_position = np.clip(new_position, bottom_limit, higher_limit)
        new_fitness = evaluate_fitness(new_position)
        if new_fitness < bee.fitness:
            bee.position = new_position
            bee.fitness = new_fitness
    return population

def onlookerBees(population, bottom_limit, higher_limit):
    population.sort(key=lambda bee: bee.fitness)
    for i in range(1, len(population)):
        phi = random.uniform(-1, 1)
        neighbor_index = random.randint(0, len(population) - 1)
        
        while neighbor_index == i:
            neighbor_index = random.randint(0, len(population) - 1)
        neighbor = population[neighbor_index]
        new_position = population[i].position + phi * (population[i].position - neighbor.position)
        new_position = np.clip(new_position, bottom_limit, higher_limit)
        new_fitness = evaluate_fitness(new_position)
        if new_fitness < population[i].fitness:
            population[i].position = new_position
            population[i].fitness = new_fitness
    return population

def random_SearchBees(population, bottom_limit, higher_limit, factor_random_search):
    for bee in population:
        if random.uniform(0, 1) < factor_random_search:
            bee.position = np.random.uniform(low=bottom_limit, high=higher_limit, size=len(bee.position))  # Dentro dos limites
            bee.fitness = evaluate_fitness(bee.position)
    return population

def ABC(problem_size, colony_size, generations, bottom_limit, higher_limit, factor_random_search):
    population = [Bee(np.random.uniform(low=bottom_limit, high=higher_limit, size=problem_size)) 
                  for _ in range(colony_size)]
    best_solution = min(population, key=lambda bee: bee.fitness)

    for generation in range(generations):
        population = employedBees(population, bottom_limit, higher_limit)
        population = onlookerBees(population, bottom_limit, higher_limit)
        population = random_SearchBees(population, bottom_limit, higher_limit, factor_random_search)

        best_solution = min(population, key=lambda bee: bee.fitness)
        print(f"Generation {generation}: Best Fitness = {best_solution.fitness}")

    return best_solution

problem_size = 2
colony_size = 100
generations = 50
factor_random_search = 0.01

#Limit of Langermann
#bottom_limit = 0
#higher_limit = 10

#Limit og DeJong N5
bottom_limit = -65.536
higher_limit = 65.536

#Limit of Schwefel 
#bottom_limit = -500
#higher_limit = 500

#Limit of Levy Function
#bottom_limit = -10
#higher_limit = 10

best_solution = ABC(problem_size, colony_size, generations, bottom_limit, higher_limit, factor_random_search)

print("Best Bee Solution:")
print("Position:", best_solution.position)
print("Fitness:", best_solution.fitness)
