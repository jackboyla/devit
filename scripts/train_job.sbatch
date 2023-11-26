#!/bin/bash
#SBATCH --job-name=devit_training
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G 
#SBATCH --gres=gpu:1 -C A100-80GB
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jboylan7@gatech.edu  # Where to send mail
#SBATCH -t480 # walltime minutes
#SBATCH -o/home/hice1/jboylan7/scratch/devit/train.log

cd /home/hice1/jboylan7/scratch/devit/

source /home/hice1/jboylan7/scratch/miniconda3/etc/profile.d/conda.sh
conda activate devit2

export vit=s
export task=ovd
export dataset=coco

# Your job command goes here
# should prepend the next command with srun but seems to cause all sorts of CPU OOM errors
bash scripts/train.sh --lora

# run this job using: SLURM_SUBMIT_DIR=/home/hice1/jboylan7/scratch/devit/scripts sbatch train_job.sbatch
# you can view logs by calling: cat ../train.log

