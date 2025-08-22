#!/bin/bash

#SBATCH --job-name=torch
#SBATCH --partition=gpu
#SBATCH --output=logs/torch_run_%j.out
#SBATCH --error=logs/torch_run_%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1        # Maximo de procesos que se usar  n
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diana.sanchez@utec.edu.pe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# module load python3/3.11.11

# source venv/bin/activate

# Nombre del script a ejecutar
script="avion_convnext.py"

# Crear carpeta para logs si no existe
mkdir -p logs

python3 $script