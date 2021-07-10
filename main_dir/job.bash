#!/bin/bash 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --qos=12c-1h_2gpu
#SBATCH --gpus=1 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=24G 
#SBATCH --job-name=”JobName” 
#SBATCH --output=JobName.%J.out 

#SBATCH --error=JobName.%J.err 
#SBATCH --mail-user=arlanvincentuy@gmail.com
#SBATCH --mail-type=ALL  
 
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST 
echo "SLURM_NNODES="$SLURM_NNODES 
echo "SLURMTMPDIR="$SLURMTMPDIR 
echo "working directory="$SLURM_SUBMIT_DIR 
 
# Place commands to load environment modules here 
module load python
 
# Set stack size to unlimited 
ulimit -s unlimited 
 
# MAIN
srun pip3 install -r requirements.txt 
srun python src/abstracts/gpu_selection.py
