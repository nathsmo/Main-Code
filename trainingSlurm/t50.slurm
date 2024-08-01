#!/bin/bash
#SBATCH --job-name=Training_TSP50_75_DPN_3G_50_en2
#SBATCH --output=outputs/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=70:00:00
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
#### TSP50

# DPN_50: Basic Pointer Network - Decoder
# python main.py --variation='DPN_50_conv' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=conv 
# python main.py --variation='DPN_50_lin' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=linear 
# python main.py --variation='DPN_50_en' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=enhanced  
# python main.py --variation='DPN_50_en2' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=enhanced2 

# DPN_3G_50: Pointer Network - Decoder + 3 Glimpses
# python main.py --variation='DPN_3G_50_conv' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=conv --n_glimpses=3 
# python main.py --variation='DPN_3G_50_lin' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=linear --n_glimpses=3
# python main.py --variation='DPN_3G_50_en' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 
# python main.py --variation='DPN_3G_50_en2' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3

# DSA_1H_50: Self-Attention Decoder (1 head)
# python main.py --variation='DSA_1H_50_conv' --task=tsp50 --n_train=75000 --decoder=self --emb_type=conv  --num_heads=1
# python main.py --variation='DSA_1H_50_lin' --task=tsp50 --n_train=75000 --decoder=self --emb_type=linear  --num_heads=1
# python main.py --variation='DSA_1H_50_en' --task=tsp50 --n_train=75000 --decoder=self --emb_type=enhanced  --num_heads=1
# python main.py --variation='DSA_1H_50_en2' --task=tsp50 --n_train=75000 --decoder=self --emb_type=enhanced2  --num_heads=1

# DSA_4H_50: Self-Attention Decoder (4 heads)
# python main.py --variation='DSA_4H_50_conv' --task=tsp50 --n_train=75000 --decoder=self --emb_type=conv  --num_heads=4
# python main.py --variation='DSA_4H_50_lin' --task=tsp50 --n_train=75000 --decoder=self --emb_type=linear  --num_heads=4
# python main.py --variation='DSA_4H_50_en' --task=tsp50 --n_train=75000 --decoder=self --emb_type=enhanced  --num_heads=4
# python main.py --variation='DSA_4H_50_en2' --task=tsp50 --n_train=75000 --decoder=self --emb_type=enhanced2  --num_heads=4

# DSA_4H_2RNN_50: Self-Attention Decoder (4 heads) + 2 RNN Layers
# python main.py --variation='DSA_4H_2RNN_50_conv' --task=tsp50 --n_train=75000 --decoder=self --emb_type=conv --rnn_layers=2 --num_heads=4
# python main.py --variation='DSA_4H_2RNN_50_lin' --task=tsp50 --n_train=75000 --decoder=self --emb_type=linear --rnn_layers=2 --num_heads=4
# python main.py --variation='DSA_4H_2RNN_50_en' --task=tsp50 --n_train=75000 --decoder=self --emb_type=enhanced --rnn_layers=2 --num_heads=4
# python main.py --variation='DSA_4H_2RNN_50_en2' --task=tsp50 --n_train=75000 --decoder=self --emb_type=enhanced2 --rnn_layers=2 --num_heads=4

echo "[$SHELL] ## Script finished"