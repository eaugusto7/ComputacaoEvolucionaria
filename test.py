import random
import numpy as np
import pandas as pd
from math import exp
from costFunctions import deJong5, langermann

def sort_func(val):
    return val[-1]

def mutate(clone, alpha, x_range):
    
    for index in range(len(clone) - 1):
        rand = np.random.normal(loc=0.0, scale=alpha, size=1)
        aux = clone[index] + rand[0]
        
        if (aux < x_range[0]):
            aux = x_range[0]

        if (aux > x_range[1]):
            aux = x_range[1]

        clone[index] = aux

    return clone        

def clonalg(cost_function, population, x_size, x_range, selection_rate, clone_rate, mutation_range):
    
    for index in range(len(population)):
        population[index][x_size] = cost_function(population[index][:-1])

    population.sort(key=sort_func)

    for i in range(round(selection_rate*len(population))):
        clones_number = round((clone_rate*len(population))/(i+1))
        fit = 1 - (i/(len(population) - 1))

        for j in range(clones_number):
            cl = population[i]
            alpha = mutation_range * exp(-fit)

            new_cl = mutate(cl, alpha, x_range)
            new_cl[-1] = cost_function(new_cl[:-1])

            if (new_cl[-1] < cl[-1]):
                population[i] = new_cl

    for i in range(round(selection_rate*len(population)), len(population)): 
        population[i] = [random.uniform(x_range[0], x_range[1]) for j in range(x_size+1)]   
        population[i][-1] = cost_function(population[i][:-1])

    return population


configs = {"config1":  [0.75, 0.8, 0.001,  100,  150 ],
           "config2":  [0.75, 0.8, 0.001,  100,  500 ],
           "config3":  [0.75, 0.8, 0.001,  300, 150 ],
           "config4":  [0.75, 0.8, 0.001,  300, 500 ],
           "config5":  [0.75, 0.8, 0.0001, 100,  150 ],
           "config6":  [0.75, 0.8, 0.0001, 100,  500 ],
           "config7":  [0.75, 0.8, 0.0001, 300, 150 ],
           "config8":  [0.75, 0.8, 0.0001, 300, 500 ],
           "config9":  [0.75, 0.9, 0.001,  100,  150 ],
           "config10": [0.75, 0.9, 0.001,  100,  500 ],
           "config11": [0.75, 0.9, 0.001,  300, 150 ],
           "config12": [0.75, 0.9, 0.001,  300, 500 ],
           "config13": [0.75, 0.9, 0.0001, 100,  150 ],
           "config14": [0.75, 0.9, 0.0001, 100,  500 ],
           "config15": [0.75, 0.9, 0.0001, 300, 150 ],
           "config16": [0.75, 0.9, 0.0001, 300, 500 ],
           "config17": [0.85, 0.8, 0.001,  100,  150 ],
           "config18": [0.85, 0.8, 0.001,  100,  500 ],
           "config19": [0.85, 0.8, 0.001,  300, 150 ],
           "config20": [0.85, 0.8, 0.001,  300, 500 ],
           "config21": [0.85, 0.8, 0.0001, 100,  150 ],
           "config22": [0.85, 0.8, 0.0001, 100,  500 ],
           "config23": [0.85, 0.8, 0.0001, 300, 150 ],
           "config24": [0.85, 0.8, 0.0001, 300, 500 ],
           "config25": [0.85, 0.9, 0.001,  100,  150 ],
           "config26": [0.85, 0.9, 0.001,  100,  500 ],
           "config27": [0.85, 0.9, 0.001,  300, 150 ],
           "config28": [0.85, 0.9, 0.001,  300, 500 ],
           "config29": [0.85, 0.9, 0.0001, 100,  150 ],
           "config30": [0.85, 0.9, 0.0001, 100,  500 ],
           "config31": [0.85, 0.9, 0.0001, 300, 150 ],
           "config32": [0.85, 0.9, 0.0001, 300, 500 ]
}


# Langermann
lang_configs_df = pd.DataFrame()

for k in configs.keys():
    
    selection_rate   = configs[k][0]
    clone_rate       = configs[k][1]
    mutation_range   = configs[k][2]
    population_size  = configs[k][3]
    number_of_epochs = configs[k][4]
    x_size = 2
    x_range = [0.0, 10.0]


    print(f"{k} -> ")


    bestOfRoundList = []
    for _ in range(30):
        print(f"{_} ")

        population = [[random.uniform(x_range[0], x_range[1]) for j in range(x_size+1)] for i in range(population_size)]

        for i in range(number_of_epochs):
            population = clonalg(langermann, population, x_size, x_range, selection_rate, clone_rate, mutation_range)
            
        population.sort(key=sort_func)
        bestOfRoundList.append(population[0][-1])

    lang_configs_df[k] = bestOfRoundList
    print("---------------------------")

lang_configs_df.to_csv("lang_configs_clonalg.csv", sep=";", index=False)

