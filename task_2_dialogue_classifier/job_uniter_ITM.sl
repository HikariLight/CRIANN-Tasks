#!/bin/bash

# Slurm submission script, 


# GPUs architecture and number
# ----------------------------
SBATCH --gpus=1
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
SBATCH --time 03:00:00
# ------------------------

module purge
module load aidl/pytorch/2.0.0-cuda11.7 
srun pip install -r requirements.txt
srun python prepare_dataset.py
srun python run_train.py --dataset_name MY_DATASET --model_name_or_path intfloat/e5-base-v2 --output_dir saved_models/MY_DATASET-e5v2-fixed_steps-smoothl1-optimized --do_train --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --evaluation_strategy steps --eval_steps 200 --max_steps 2000 --warmup_ratio 0.1 --logging_strategy steps --logging_steps 25 --save_strategy steps --save_steps 200 --seed 0 --data_seed 0 --word_encoder_layers_to_train ".*attention.*$" --phrase_hidden_size 768 --phrase_intermediate_size 1536 --loss_reduction mean --optim adamw_torch --learning_rate 0.00002