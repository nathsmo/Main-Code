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
# python main.py --task=tsp50 --n_train=35000
# Basic + Self attention
# python main.py --task=tsp10 --decoder=self --num_heads=4 --n_train=100000
# python main.py --task=tsp20 --decoder=self --num_heads=4 --n_train=100000
# python main.py --task=tsp50 --decoder=self --num_heads=4 --n_train=35000
# Basic training + Embedding Enhanced
# python main.py --task=tsp10 --n_train=100000 --emb_type=enhanced
# python main.py --task=tsp20 --n_train=100000 --emb_type=enhanced
# python main.py --task=tsp50 --n_train=35000 --emb_type=enhanced
# Basic
# python main.py --task=tsp10 --n_train=100000 --n_glimpses=3
# python main.py --task=tsp20 --n_train=100000 --n_glimpses=3
# python main.py --task=tsp50 --n_train=35000 --n_glimpses=3


##### Evaluation
# python main.py --task=tsp10 --is_train=False --emb_type=enhanced --model_dir=./output_models/agent_complete.pth
# python main.py --task=tsp10 --is_train=False --emb_type=enhanced --infer_type=single --model_dir=./output_models/agent_complete.pth



# Training organized

# A1: Basic
# python main.py --variation='A1' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear 

# B1: Decoder change + Basic
# python main.py --variation='B1' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1

# C1: Emb type change + Basic
# python main.py --variation='C1' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced  

# D1: New combination
# python main.py --variation='D1' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced --num_heads=4

# E1: Enhanced emb + Glimpse
# python main.py --variation='E1' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 

# F1: New combination + rnn
# python main.py --variation='F1' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --rnn_layers=2 --num_heads=4

#### TSP20

# A2: Basic
# python main.py --variation='A2' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=linear  

# B2: Decoder change + Basic
# python main.py --variation='B2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1

# C2: Emb type change + Basic
# python main.py --variation='C2' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced  

# # D2: New combination
# python main.py --variation='D2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=4

# # E2: Enhanced emb + Glimpse
# python main.py --variation='E2' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 

# # F2: New combination + rnn
# python main.py --variation='F2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced  --rnn_layers=2 --num_heads=4


echo "[$SHELL] ## Script finished"