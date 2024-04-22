#!/bin/bash

# Nome do arquivo de saída
output_file="Conf1_DeJong.csv"

for i in {1..30}; do
    # Executa o programa e redireciona a saída para o arquivo de saída
    #go run GeneticAlgorithm.go >> "$output_file" 2>&1
    python3 ParticleSwarmOptimization.py >> "$output_file" 2>&1
    echo "Execução $i concluída"
done
