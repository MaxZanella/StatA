#!/bin/bash

# Paths and architecture
DATA=$1
ARCH=$2

# Datasets
datasets=("imagenet" "sun397" "fgvc" "eurosat" "stanford_cars" "food101" "oxford_pets" "oxford_flowers" "caltech101" "dtd" "ucf101")

# Number of tasks
n_tasks=100

# Gamma values
gamma_values=(0.1 0.01 0.001 -1)

# Loop over batch sizes and configurations
for batch_size in 128; do
  for dataset in "${datasets[@]}"; do
    for gamma in "${gamma_values[@]}"; do
      python3 main.py --root_path "${DATA}" \
                      --dataset "$dataset" \
                      --method StatA \
                      --backbone "${ARCH}" \
                      --batch_size "$batch_size" \
                      --online \
                      --gamma "$gamma" \
                      --n_tasks "$n_tasks"
    done
  done
done
