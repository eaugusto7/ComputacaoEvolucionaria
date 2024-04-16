#!/bin/bash

# Nome do arquivo de saída
output_file="Conf30_Torneio_Langermann.csv"

for i in {1..30}; do
    # Executa o programa e redireciona a saída para o arquivo de saída
    go run Langermann_Function.go >> "$output_file" 2>&1
    echo "Execução $i concluída"
done

#output_lines=$(tail -n 30 "$output_file")
#fitness_values=$(echo "$output_lines" | awk '{print $NF}')

# Calcular média, mediana, valor máximo e valor mínimo
#average=$(echo "$fitness_values" | awk '{sum+=$1} END {print sum/NR}')
#median=$(echo "$fitness_values" | sort -n | awk ' { a[i++]=$1; } END { print a[int(i/2)]; }')
#max_value=$(echo "$fitness_values" | sort -g | tail -n 1)
#min_value=$(echo "$fitness_values" | sort -n | head -n 1)

# Adicionar os resultados ao final do arquivo de saída
#echo "Média: $average" >> "$output_file"
#echo "Mediana: $median" >> "$output_file"
#echo "Valor Máximo: $max_value" >> "$output_file"
#echo "Valor Mínimo: $min_value" >> "$output_file"