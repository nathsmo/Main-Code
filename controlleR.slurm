#!/bin/bash
#SBATCH --job-name=main_code
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=1T
#SBATCH --time=24:00:00
#SBATCH --partition=amd-gpu-long

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python/3.9.5-GCCcore-10.3.0

# Source the Python virtual environment
source $HOME/user_guide_tutorials/first_python_job/python_test_venv/bin/activate

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 main.py --task=tsp10

echo "[$SHELL] ## Script finished"