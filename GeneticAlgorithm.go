package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

// Define a estrutura para um indivíduo na população
type Individual struct {
	Chromosome []int
	Fitness    float64
}

// Função de Langermann
func langermann(xx []float64, dimensionAdjustment float64) float64 {
	d := len(xx)
	m := 5
	c := []float64{1, 2, 5, 2, 3}
	A := [][]float64{{3, 5}, {5, 2}, {2, 1}, {1, 4}, {7, 9}}

	outer := 0.0
	for i := 0; i < m; i++ {
		inner := 0.0
		for j := 0; j < d; j++ {
			inner += math.Pow(xx[j]-A[i][j], 2)
		}
		newTerm := c[i] * math.Exp(-inner/math.Pi) * math.Cos(math.Pi*inner)
		outer += newTerm
	}
	fmt.Println(xx)
	fmt.Println(outer)
	return outer + dimensionAdjustment
}

func dejong5(xx []float64, dimensionAdjustment float64) float64 {
	outer := 0.0

	a1 := []float64{-32.0, -16.0, 0.0, 16.0, 32.0,
		-32.0, -16.0, 0.0, 16.0, 32.0,
		-32.0, -16.0, 0.0, 16.0, 32.0,
		-32.0, -16.0, 0.0, 16.0, 32.0,
		-32.0, -16.0, 0.0, 16.0, 32.0}

	a2 := []float64{-32.0, -32.0, -32.0, -32.0, -32.0,
		-16.0, -16.0, -16.0, -16.0, -16.0,
		0.0, 0.0, 0.0, 0.0, 0.0,
		16.0, 16.0, 16.0, 16.0, 16.0,
		32.0, 32.0, 32.0, 32.0, 32.0}

	sum := 0.002
	for i := 1; i < 26; i++ {
		sum += (1 / (float64(i) + math.Pow((xx[0]-a1[i-1]), 6) + math.Pow((xx[1]-a2[i-1]), 6)))
	}

	outer = 1.0 / sum

	return outer + dimensionAdjustment
}

// Função para inicializar a população
func initializePopulation(populationSize, chromosomeLength int) []Individual {
	population := make([]Individual, populationSize)
	for i := range population {
		chromosome := make([]int, chromosomeLength)
		for j := range chromosome {
			chromosome[j] = rand.Intn(2)
		}
		population[i] = Individual{Chromosome: chromosome}
	}
	return population
}

// Função para avaliar a população
func evaluatePopulation(population []Individual, dimensionAdjustment float64) {
	for i := range population {
		x, y := decodeChromosome(population[i].Chromosome)
		population[i].Fitness = langermann([]float64{x, y}, dimensionAdjustment)
		//population[i].Fitness = dejong5([]float64{x, y}, dimensionAdjustment)
	}
}

// Função para decodificar o cromossomo em valores reais
func decodeChromosome(chromosome []int) (float64, float64) {
	x := 0.0
	y := 0.0
	for i, gene := range chromosome {
		x += float64(gene) * math.Pow(2, float64(i))
		y += float64(gene) * math.Pow(2, float64(len(chromosome)-i-1))
	}
	limitSup := 10.0
	limitInf := 0.0

	//limitSup := 65.536
	//limitInf := -65.536

	precisao := (limitSup - limitInf) / (math.Pow(2, float64(len(chromosome))) - 1)

	x = limitInf + precisao*x
	y = limitInf + precisao*y

	return x, y
}

// Função para seleção de pais por roleta viciada
func rouletteSelection(population []Individual) []Individual {
	parents := make([]Individual, len(population))
	fitnessSum := 0.0

	// Calcular a soma total de fitness
	for _, individual := range population {
		fitnessSum += 1 / individual.Fitness
	}

	// Selecionar pais com base na roleta viciada
	for i := range parents {
		r := rand.Float64() * fitnessSum
		var sum float64
		for _, individual := range population {
			sum += 1 / individual.Fitness
			if sum >= r {
				parents[i] = individual
				break
			}
		}
	}

	return parents
}

// Função para seleção de pais por torneio
func tournamentSelection(population []Individual, tournamentSize int) []Individual {
	parents := make([]Individual, len(population))
	var competitors_list1 []Individual
	var competitors_list2 []Individual

	for i := 0; i < tournamentSize; i++ {
		competitorIndex := rand.Intn(len(population))
		competitors_list1 = append(competitors_list1, population[competitorIndex])
	}

	for i := 0; i < tournamentSize; i++ {
		competitorIndex := rand.Intn(len(population))
		competitors_list2 = append(competitors_list2, population[competitorIndex])
	}

	for j := 0; j < len(competitors_list1); j++ {
		if competitors_list1[j].Fitness <= competitors_list2[j].Fitness {
			parents[j] = competitors_list1[j]
		} else {
			parents[j] = competitors_list2[j]
		}
	}
	return parents
}

// Função de cruzamento com um ponto de corte por variável
func crossover(parent1, parent2 Individual, crossoverRate float64) (Individual, Individual) {
	child1 := Individual{Chromosome: make([]int, len(parent1.Chromosome))}
	child2 := Individual{Chromosome: make([]int, len(parent1.Chromosome))}
	crossoverPoint := rand.Intn(len(parent1.Chromosome))
	crossoverPoint2 := rand.Intn(len(parent1.Chromosome))

	if rand.Float64() < crossoverRate {
		for i := 0; i < crossoverPoint/2; i++ {
			child1.Chromosome[i] = parent1.Chromosome[i]
			child2.Chromosome[i] = parent2.Chromosome[i]
		}

		for i := crossoverPoint; i < len(parent1.Chromosome); i++ {
			child1.Chromosome[i] = parent2.Chromosome[i]
			child2.Chromosome[i] = parent1.Chromosome[i]
		}

		for i := crossoverPoint / 2; i < crossoverPoint2; i++ {
			child1.Chromosome[i] = parent1.Chromosome[i]
			child2.Chromosome[i] = parent2.Chromosome[i]
		}

		for i := crossoverPoint2; i < len(parent1.Chromosome); i++ {
			child1.Chromosome[i] = parent2.Chromosome[i]
			child2.Chromosome[i] = parent1.Chromosome[i]
		}
	} else {
		child1.Chromosome = parent1.Chromosome
		child2.Chromosome = parent2.Chromosome
	}
	return child1, child2
}

// Função de mutação bit a bit
func mutate(child Individual, mutationRate float64) Individual {
	for i := range child.Chromosome {
		if rand.Float64() < mutationRate {
			// Flip the bit with the mutation probability
			if child.Chromosome[i] == 0 {
				child.Chromosome[i] = 1
			} else {
				child.Chromosome[i] = 0
			}
		}
	}
	return child
}

// Algoritmo genético principal
func geneticAlgorithm(populationSize, chromosomeLength, generations int, crossoverRate, mutationRate float64, selectionMethod string, elitism bool, dimensionAdjustment float64) Individual {
	population := initializePopulation(populationSize, chromosomeLength)
	//bestPrevious := Individual{}

	for generation := 0; generation < generations; generation++ {
		evaluatePopulation(population, dimensionAdjustment)
		sort.Slice(population, func(i, j int) bool {
			return population[i].Fitness < population[j].Fitness
		})

		parents := make([]Individual, populationSize)
		if selectionMethod == "roulette" {
			parents = rouletteSelection(population)
		} else if selectionMethod == "tournament" {
			parents = tournamentSelection(population, populationSize)
		}

		newPopulation := make([]Individual, populationSize)

		if elitism {
			newPopulation[0] = population[0]
			newPopulation[1] = population[1]
		}

		for i := 0; i < len(parents); i += 2 {
			if elitism && i == 0 {
				continue
			}
			child1, child2 := crossover(parents[i], parents[i+1], crossoverRate)
			child1 = mutate(child1, mutationRate)
			child2 = mutate(child2, mutationRate)
			newPopulation[i], newPopulation[i+1] = child1, child2
		}
		fmt.Println(population[0].Fitness - dimensionAdjustment)

		totalFitness := 0.0
		for _, individual := range population {
			totalFitness += (individual.Fitness - dimensionAdjustment)
			//fmt.Println(individual.Fitness - dimensionAdjustment)
		}
		fitnessMedio := totalFitness / float64(len(population))
		fmt.Println(fitnessMedio)

		if (generation + 1) != generations {
			population = newPopulation
		}
	}
	return population[0] // Retorna o melhor indivíduo após todas as gerações
}

func main() {
	rand.Seed(time.Now().UnixNano())

	populationSize := 80
	chromosomeLength := 600
	generations := 150
	crossoverRate := 0.9
	mutationRate := 0.03
	selectionMethod := "roulette" // Pode ser "roulette" ou "tournament"
	elitism := true               // Define se o elitismo será aplicado
	dimensionAdjustment := 100.0

	bestIndividual := geneticAlgorithm(populationSize, chromosomeLength, generations, crossoverRate, mutationRate, selectionMethod, elitism, dimensionAdjustment)
	//fmt.Println("\nMelhor indivíduo:", bestIndividual)
	fmt.Println("Fitness Individuo Final")
	fmt.Println(bestIndividual.Fitness - dimensionAdjustment)

	//vector := []float64{1.0, 2.0}
	//fmt.Print(dejong5(vector))

}
