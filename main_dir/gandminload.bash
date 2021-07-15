#!/bin/bash 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=12c-1h_2gpu
#SBATCH --gpus=1
  
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=24G 
#SBATCH --job-name=”gandminload” 
#SBATCH --output=gandminload.%J.out 

#SBATCH --error=gandminload.%J.err 
#SBATCH --mail-type=ALL
#SBATCH --requeue 
 
echo "SLURM_JOBID="$SLURM_JOBID 
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST 
echo "SLURM_NNODES="$SLURM_NNODES 
echo "SLURMTMPDIR="$SLURMTMPDIR 
echo "working directory="$SLURM_SUBMIT_DIR 
 
# Place commands to load environment modules here 
module load anaconda/3-5.3.1
module load python 
module load cuda/10.1_cudnn-7.6.5
 
# Set stack size to unlimited 
ulimit -s unlimited 
 
# MAIN 
srun python -m pip install --user --upgrade pip
srun python -m pip install --user -r requirements.txt
srun python main.py 2 2 10 cpuandminimal00.yaml gpuandminimal00.yaml
