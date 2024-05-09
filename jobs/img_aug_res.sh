     Show Dotfiles Show Owner/Mode
/scrfs/storage/jdivers/home/ondemand/data/sys/myjobs/projects/default/12/
#!/bin/bash

#SBATCH -J nsclc_img_aug_res
#SBATCH --partition gpu72
#SBATCH -o nsclc_img_aug_res.txt
#SBATCH -e nsclc_img_aug_res.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdivers@uark.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=72:00:00

export OMP_NUM_THREADS=32

# load required module
module purge
module load python

cd $SLURM_SUBMIT_DIR || exit
# input files needed for job
files=/home/jdivers/ondemand/data/sys/myjobs/projects/default/6/nsclc_project/data
rsync -av -q $files /scratch/$SLURM_JOB_ID
rsync -av -q my_modules /scratch/$SLURM_JOB_ID
rsync -av -q myenv /scratch/$SLURM_JOB_ID
rsync -av -q *.py /scratch/$SLURM_JOB_ID
wait

cd /scratch/$SLURM_JOB_ID/ || exit

# Init and activate venv
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install torchvision
pip install opencv-python
pip install openpyxl

# Run main
python3 image_augmented_response.py

rsync -av -q /scratch/$SLURM_JOB_ID/ $SLURM_SUBMIT_DIR