#!/bin/bash
#SBATCH --qos=240c-1h_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24G
#SBATCH --job-name=”rv1”
#SBATCH --output=JobName.%J.out

#SBATCH --error=JobName.%J.err
#SBATCH --mail-user=gnhs.com@gmail.com
#SBATCH --mail-type=ALL

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Place commands to load environment modules here

# Set stack size to unlimited

ulimit -s unlimited

# MAIN
srun python3 -m pip install --upgrade pip --user
srun python3 -m pip install -r requirements.txt
srun python3 src/abstracts/gpu_selection.py
