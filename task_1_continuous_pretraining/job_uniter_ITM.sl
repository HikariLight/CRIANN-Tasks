#!/bin/bash

# Slurm submission script, 


# GPUs architecture and number
# ----------------------------
SBATCH --gpus=2
# ------------------------

# processes / tasks
## nombre de noeud voulu
SBATCH -n 1
# ------------------------

# ----------------------------
# CPUs per task
SBATCH --cpus-per-task 8
# ------------------------

# ------------------------
# Job time (hh:mm:ss)
SBATCH --time 24:00:00
# ------------------------

module purge
module load aidl/pytorch/2.0.0-cuda11.7 
srun pip install -r requirements.txt
srun python train.py
