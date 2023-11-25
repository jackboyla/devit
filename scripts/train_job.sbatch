#!/bin/bash
#SBATCH --job-name=devit_training
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1 -C V100-32GB
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<your-email>  # Where to send mail
#SBATCH -t480 # walltime minutes
#SBATCH -o/devit/train.log

cd /devit/

source /home/miniconda3/etc/profile.d/conda.sh
conda activate devit2

export vit=s
export task=ovd
export dataset=coco

# Your job command goes here
srun bash scripts/train.sh --lora

# run this job using: SLURM_SUBMIT_DIR=/devit/scripts sbatch train_job.sbatch
