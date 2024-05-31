#!/bin/bash

#SBATCH --job-name=big-comet_iterator
#SBATCH --partition=agpu06
#SBATCH --output=nsclc_main.txt
#SBATCH --error=nsclc_main.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdivers@uark.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --qos=gpu

export OMP_NUM_THREADS=64

# load required module
module purge
module load python/anaconda-3.14

# Activate venv
conda activate /home/jdivers/.conda/envs/dl_env
echo $SLURM_JOB_ID

cd $SLURM_SUBMIT_DIR || exit
# input files needed for job
files=/home/jdivers/ondemand/data/sys/myjobs/projects/nsclc/data

echo "Copying files..."
rsync -avq $files /scratch/$SLURM_JOB_ID
rsync -avq $SLURM_SUBMIT_DIR/big_comet.py /scratch/$SLURM_JOB_ID
rsync -avq /home/jdivers/nsclc/my_modules /scratch/$SLURM_JOB_ID
wait

cd /scratch/$SLURM_JOB_ID/ || exit

echo "Python script initiating..."
python3 big_comet.py

rsync -avq /scratch/$SLURM_JOB_ID/outputs $SLURM_SUBMIT_DIR

# check if rsync succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync files back to original directory. Check /scratch/$SLURM_JOB_ID/ for output files."
  exit 1
fi
