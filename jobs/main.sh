#!/bin/bash

#SBATCH --job-name=nsclc_main
#SBATCH --partition=qgpu06
#SBATCH --output=nsclc_main.txt
#SBATCH --error=nsclc_main.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdivers@uark.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --qos=gpu

export OMP_NUM_THREADS=1

# load required module
module purge
module load python/anaconda-3.14

# Activate venv
conda activate /home/jdivers/.conda/envs/dl_env

cd $SLURM_SUBMIT_DIR || exit
# input files needed for job
files=/home/jdivers/ondemand/data/sys/myjobs/projects/nsclc/data
rsync -av -q $files /scratch/$SLURM_JOB_ID
wait

cd /scratch/$SLURM_JOB_ID/ || exit

python3 main.py

mkdir -p $SLURM_SUBMIT_DIR/$job_name
rsync -av -q /scratch/$SLURM_JOB_ID/ $SLURM_SUBMIT_DIR/$job_name

# check if rsync succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync files back to original directory. Check /scratch/$SLURM_JOB_ID/ for output files."
  exit 1
