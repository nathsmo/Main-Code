#!/bin/bash
#SBATCH --job-name=Training_TSP100_75_DSA_4H_100_en2
#SBATCH --output=outputs/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=99:00:00
#SBATCH --partition=mem
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

# Training organized
#### TSP100

# DPN_100: Basic Pointer Network - Decoder
# python main.py --variation='DPN_100_conv' --task=tsp100 --n_train=75000 --decoder=pointer --emb_type=conv 
# python main.py --variation='DPN_100_lin' --task=tsp100 --n_train=75000 --decoder=pointer --emb_type=linear 
# python main.py --variation='DPN_100_en' --task=tsp100 --n_train=75000 --decoder=pointer --emb_type=enhanced  
# python main.py --variation='DPN_100_en2' --task=tsp100 --n_train=75000 --decoder=pointer --emb_type=enhanced2 

# DSA_1H_100: Self-Attention Decoder (1 head)
# python main.py --variation='DSA_1H_100_conv' --task=tsp100 --n_train=75000 --decoder=self --emb_type=conv  --num_heads=1
# python main.py --variation='DSA_1H_100_lin' --task=tsp100 --n_train=75000 --decoder=self --emb_type=linear  --num_heads=1
# python main.py --variation='DSA_1H_100_en' --task=tsp100 --n_train=75000 --decoder=self --emb_type=enhanced  --num_heads=1
# python main.py --variation='DSA_1H_100_en2' --task=tsp100 --n_train=75000 --decoder=self --emb_type=enhanced2  --num_heads=1

# DSA_4H_100: Self-Attention Decoder (4 heads)
# python main.py --variation='DSA_4H_100_conv' --task=tsp100 --n_train=75000 --decoder=self --emb_type=conv  --num_heads=4
# python main.py --variation='DSA_4H_100_lin' --task=tsp100 --n_train=75000 --decoder=self --emb_type=linear  --num_heads=4
# python main.py --variation='DSA_4H_100_en' --task=tsp100 --n_train=75000 --decoder=self --emb_type=enhanced  --num_heads=4
python main.py --variation='DSA_4H_100_en2' --task=tsp100 --n_train=75000 --decoder=self --emb_type=enhanced2  --num_heads=4

# DSA_4H_2RNN_100: Self-Attention Decoder (4 heads) + 2 RNN Layers
# python main.py --variation='DSA_4H_2RNN_100_conv' --task=tsp100 --n_train=75000 --decoder=self --emb_type=conv --rnn_layers=2 --num_heads=4
# python main.py --variation='DSA_4H_2RNN_100_lin' --task=tsp100 --n_train=75000 --decoder=self --emb_type=linear --rnn_layers=2 --num_heads=4
# python main.py --variation='DSA_4H_2RNN_100_en' --task=tsp100 --n_train=75000 --decoder=self --emb_type=enhanced --rnn_layers=2 --num_heads=4
# python main.py --variation='DSA_4H_2RNN_100_en2' --task=tsp100 --n_train=75000 --decoder=self --emb_type=enhanced2 --rnn_layers=2 --num_heads=4

echo "[$SHELL] ## Script finished"