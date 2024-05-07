import random
import numpy as np
import math
import matplotlib.pyplot as plt
class Bee:
    def __init__(self, position):
        self.position = position
        self.fitness = evaluate_fitness(position)

def ackley(xx, a=20, b=0.2, c=2*np.pi):
    d = len(xx)

    sum1 = np.sum(np.fromiter((xi**2 for xi in xx), dtype=float))
    sum2 = np.sum(np.fromiter((np.cos(c * xi) for xi in xx), dtype=float))

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    y = term1 + term2 + a + np.exp(1)

    return y
    
def bukin6(xx):
    x1, x2 = xx

    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
    term2 = 0.01 * np.abs(x1 + 10)

    y = term1 + term2

    return y

def crossit(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(abs(100 - np.sqrt(x1**2 + x2**2)) / np.pi)

    y = -0.0001 * (abs(fact1 * fact2) + 1) ** 0.1

    return y

def drop(xx):
    x1, x2 = xx
    
    frac1 = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    frac2 = 0.5 * (x1**2 + x2**2) + 2

    y = -frac1 / frac2
    
    return y


def egg(xx):
    x1, x2 = xx
    
    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

    y = term1 + term2
    
    return y

def griewank(xx):
    d = len(xx)
    _sum = 0
    _prod = 1

    for ii in range(d):
        xi = xx[ii]
        _sum += xi ** 2 / 4000
        _prod *= np.cos(xi / np.sqrt(ii + 1))

    y = _sum - _prod + 1
    
    return y

def holder(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1 = np.sin(x1) * np.cos(x2)
    fact2 = np.exp(abs(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))

    y = -abs(fact1 * fact2)

    return y

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

def levy13(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = (np.sin(3 * np.pi * x1)) ** 2
    term2 = (x1 - 1) ** 2 * (1 + (np.sin(3 * np.pi * x2)) ** 2)
    term3 = (x2 - 1) ** 2 * (1 + (np.sin(2 * np.pi * x2)) ** 2)

    y = term1 + term2 + term3

    return y

def rastrigin(xx):
    d = len(xx)
    _sum = 0
    for xi in xx:
        _sum += (xi ** 2 - 10 * np.cos(2 * np.pi * xi))
    
    y = 10 * d + _sum
    return y

def schaffer2(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1 = (np.sin(x1**2 - x2**2))**2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2))**2

    y = 0.5 + fact1 / fact2

    return y

def schaffer4(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1 = (np.cos(np.sin(abs(x1**2 - x2**2))))**2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2))**2

    y = 0.5 + fact1 / fact2

    return y

def shubert(xx):
    x1 = xx[0]
    x2 = xx[1]
    sum1 = 0
    sum2 = 0

    for ii in range(1, 6):
        new1 = ii * np.cos((ii + 1) * x1 + ii)
        new2 = ii * np.cos((ii + 1) * x2 + ii)
        sum1 += new1
        sum2 += new2

    y = sum1 * sum2

    return y

def easom(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1 = -np.cos(x1) * np.cos(x2)
    fact2 = np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)
    y = fact1 * fact2
    return y

def michal(xx, m=10):
    d = len(xx)
    _sum = 0

    for ii in range(d):
        xi = xx[ii]
        new = np.sin(xi) * (np.sin((ii + 1) * xi**2 / np.pi))**(2 * m)
        _sum += new

    y = -_sum
    return y

def beale(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = (1.5 - x1 + x1 * x2) ** 2
    term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
    term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2

    y = term1 + term2 + term3

    return y

def branin(xx, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    x1 = xx[0]
    x2 = xx[1]

    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)

    y = term1 + term2 + s

    return y

def goldpr(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1a = (x1 + x2 + 1) ** 2
    fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2 * x1 - 3 * x2) ** 2
    fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
    fact2 = 30 + fact2a * fact2b

    y = fact1 * fact2

    return y

def permdb(xx, b=0.5):
    d = len(xx)
    outer = 0

    for ii in range(1, d+1):
        inner = 0
        for jj in range(1, d+1):
            xj = xx[jj-1]
            inner += (jj ** ii + b) * ((xj / jj) ** ii - 1)
        outer += inner ** 2

    y = outer
    return y

def powell(xx):
    x1, x2 = xx
    term1 = (x1 + 10*x2) ** 2
    term2 = 5 * (x1 - x2) ** 2
    term3 = (x1 - 2*x2) ** 4
    term4 = 10 * (x2 - x1) ** 4
    y = term1 + term2 + term3 + term4
    return y

def stybtang(xx):
    x1, x2 = xx
    term1 = x1**4 - 16*x1**2 + 5*x1
    term2 = x2**4 - 16*x2**2 + 5*x2
    y = (term1 + term2) / 2
    return y

def evaluate_fitness(position):
    #return ackley(position)
    #return bukin6(position) #Revisar essa funcao
    #return crossit(position)
    #return drop(position)
    #return egg(position)
    #return griewank(position)
    #return holder(position)
    #return langermann(position)
    #return deJong5(position)
    #return schwef(position)
    #return levy(position)
    #return levy13(position)
    #return rastrigin(position)
    #return schaffer2(position)
    #return schaffer4(position)
    #return shubert(position)
    #return easom(position)
    #return michal(position)
    #return beale(position)
    #return branin(position)
    #return goldpr(position)
    #return permdb(position)
    #return powell(position)
    return stybtang(position)

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

def scoutBees(population, bottom_limit, higher_limit, factor_random_search):
    for bee in population:
        if random.uniform(0, 1) < factor_random_search:
            bee.position = np.random.uniform(low=bottom_limit, high=higher_limit, size=len(bee.position))
            bee.fitness = evaluate_fitness(bee.position)
    return population

def ABC(problem_size, colony_size, generations, bottom_limit, higher_limit, factor_random_search):
    population = [Bee(np.random.uniform(low=bottom_limit, high=higher_limit, size=problem_size)) 
                  for _ in range(colony_size)]
    best_solution = min(population, key=lambda bee: bee.fitness)

    for generation in range(generations):
        population = employedBees(population, bottom_limit, higher_limit)
        population = onlookerBees(population, bottom_limit, higher_limit)
        population = scoutBees(population, bottom_limit, higher_limit, factor_random_search)

        best_solution = min(population, key=lambda bee: bee.fitness)
        print(f"Generation {generation}: Best Fitness = {best_solution.fitness}")

    return best_solution

problem_size = 2
colony_size = 150
generations = 80
factor_random_search = 0.01

#Limit of Ackley
#bottom_limit = -32.768
#higher_limit = 32.768

#Revisar essa funcao
#Limit of Bukin6
#bottom_limit = -15.0
#higher_limit = -5.0

#Limit of Crossfit
#bottom_limit = -10.0
#higher_limit = 10.0

#Limit of Drop
#bottom_limit = -5.12
#higher_limit = 5.12

#Limit of Egg
#bottom_limit = -512.0
#higher_limit = 512.0

#Limit of Griewank
#bottom_limit = -600.0
#higher_limit = 600.0

#Limit of Holder
#bottom_limit = -10.0
#higher_limit = 10.0

#Limit of Langermann
#bottom_limit = 0
#higher_limit = 10

#Limit og DeJong N5
#bottom_limit = -65.536
#higher_limit = 65.536

#Limit of Schwefel 
#bottom_limit = -500
#higher_limit = 500

#Limit of Levy
#bottom_limit = -10
#higher_limit = 10

#Limit of Rastrigin
#bottom_limit = -5.12
#higher_limit = 5.12

#Limit of Schaffer
#bottom_limit = -100.0
#higher_limit = 100.0

#Limit of Shubert
#bottom_limit = -10.0
#higher_limit = 10.0

#Limit of Easom
#bottom_limit = -100.0
#higher_limit = 100.0

#Limit of Michalewicz
#bottom_limit = 0.0
#higher_limit = math.pi

#Limit of Beale
#bottom_limit = -4.5
#higher_limit = 4.5

#Limit of Branin
#bottom_limit = 0
#higher_limit = 15.0

#Limit of GoldPr
#bottom_limit = -2.0
#higher_limit = 2.0

#Limit of PermDb
#bottom_limit = -2.0
#higher_limit = 2.0

#Limit of Powell
#bottom_limit = -4.0
#higher_limit = 5.0

#Limit of Styblinski-Tang
bottom_limit = -5.0
higher_limit = 5.0

best_solution = ABC(problem_size, colony_size, generations, bottom_limit, higher_limit, factor_random_search)

#print("Best Bee Solution:")
#print("Position:", best_solution.position)
#print("Fitness:", best_solution.fitness)

print(best_solution.fitness)

# Realizar 30 execuções e guardar os resultados
#results = []
#for _ in range(30):
#    result = ABC(problem_size, colony_size, generations, bottom_limit, higher_limit, factor_random_search).fitness
#    results.append(result)

# Calcular estatísticas
#mean_result = np.mean(results)
#median_result = np.median(results)
#max_result = np.max(results)
#min_result = np.min(results)

# Plotar os resultados
#plt.figure(figsize=(8, 6))
#plt.hist(results, bins=10, edgecolor='black')
#plt.axvline(x=mean_result, color='r', linestyle='--', label=f'Mean: {mean_result:.2f}')
#plt.axvline(x=median_result, color='g', linestyle='--', label=f'Median: {median_result:.2f}')
#plt.axvline(x=max_result, color='b', linestyle='--', label=f'Max: {max_result:.2f}')
#plt.axvline(x=min_result, color='y', linestyle='--', label=f'Min: {min_result:.2f}')
#plt.xlabel('Fitness')
#plt.ylabel('Frequency')
#plt.title('Histogram of Fitness Values')
#plt.legend()
#plt.grid(True)
#plt.show()
