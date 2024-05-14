#!/bin/bash

#Init empty script variable
py_script=""

# Parse command-line args
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -n|--name) job_name="$2"; shift ;;
    -p|--partition) partition="$2"; shift ;;
    -wt|--walltime) walltime="$2"; shift ;;
    -nt|--numtasks) numtasks="$2"; shift ;;
    -s|--script) py_script="$2"; shift ;;
    -v|--venv) venv_path="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
done

# Defaults LUT
declare -A lut
lut["comp01"]=("comp" "01:00:00" "1" "32")
lut["gpu72"]=("gpu" "72:00:00" "32" "1")
lut["agpu06"]=("gpu" "06:00:00" "64" "1")

# Check for script input
if [ -z "$py_script" ]; then
  echo "Error: No python script provided. Please specify a script using the -s or --script flag."
  exit 1
fi

# Set defaults
job_name="${job_name:-my_job}"
partition="${partition:-comp01}"
walltime="${walltime:-"${lut["$partition"][1]}"}"
numtasks="${numtasks:-"${lut["$partition"][2]}"}"
cpus_per="${cpus_per:-"${lut["$partition"][3]}"}"
qos="${}lut["$partition"][0]}"


#SBATCH --job-name="$job_name"
#SBATCH --partition="$partition"
#SBATCH --output="$job_name.txt"
#SBATCH --error="$job_name.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdivers@uark.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node="$numtasks"
#SBATCH --cpus-per-task="$cpus_per"
#SBATCH --time="$walltime"
#SBATCH --qos="$qos"

export OMP_NUM_THREADS="$cpus_per"

# load required module
module purge
module load python/anaconda-3.14

# Activate venv
conda activate "$venv_path"

cd $SLURM_SUBMIT_DIR || exit
# input files needed for job
files=/home/jdivers/ondemand/data/sys/myjobs/projects/nsclc/data
rsync -av -q $files /scratch/$SLURM_JOB_ID

# remove any outdated clones and clone remote into submitdir
rm -rf nsclc
git clone --depth 1 https://github.com/jiversivers/nsclc.git

# Copy program files to scratch
rsync -av -q nsclc/my_modules /scratch/$SLURM_JOB_ID
rsync -av -q nsclc/jobs/*.py /scratch/$SLURM_JOB_ID
wait

cd /scratch/$SLURM_JOB_ID/ || exit

python3 "$py_script"

mkdir -p $SLURM_SUBMIT_DIR/$job_name
rsync -av -q /scratch/$SLURM_JOB_ID/ $SLURM_SUBMIT_DIR/$job_name

# check if rsync succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync files back to original directory. Check /scratch/$SLURM_JOB_ID/ for output files."
  exit 1
fi