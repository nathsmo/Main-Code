#!/bin/bash
#SBATCH --job-name=main_code
#SBATCH --output=outputs/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=40:00:00
#SBATCH --partition=amd-gpu-long
#SBATCH --ntasks=1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python/3.11.3-GCCcore-12.3.0
module load Miniconda3/23.9.0-0

# Source the Python virtual environment
# source $HOME/user_guide_tutorials/thesis/torch_thesis_server/bin/activate

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
# Basic training
# python main.py --task=tsp10 --n_train=100000 
# python main.py --task=tsp20 --n_train=100000
# python main.py --task=tsp50 --n_train=10000
# Basic + Self attention
# python main.py --task=tsp10 --decoder=self --num_heads=4 --n_train=100000
# python main.py --task=tsp20 --decoder=self --num_heads=4 --n_train=100000
# python main.py --task=tsp50 --decoder=self --num_heads=4 --n_train=10000
# Basic training + Embedding Enhanced
# python main.py --task=tsp10 --n_train=100000 --emb_type=enhanced
# python main.py --task=tsp20 --n_train=100000 --emb_type=enhanced
# python main.py --task=tsp50 --n_train=10000 --emb_type=enhanced
# Basic
# python main.py --task=tsp10 --n_train=100000 --n_glimpses=3
# python main.py --task=tsp20 --n_train=100000 --n_glimpses=3
# python main.py --task=tsp50 --n_train=10000 --n_glimpses=3

echo "[$SHELL] ## Script finished"