#!/bin/bash
#SBATCH --job-name=jack_job
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2 -C A100-40GB
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<you-email-here>  # Where to send mail
#SBATCH -t480 # walltime minutes
#SBATCH -o/<output.log>

conda activate devit2

# Your job command goes here
srun vit=l task=ovd dataset=coco bash train.sh

# run this job using: $SLURM_SUBMIT_DIR=<path-to-this-file> sbatch train_job.sbatch
