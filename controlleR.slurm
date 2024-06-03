#!/bin/bash
#SBATCH --job-name=Eval_B1
#SBATCH --output=outputs/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=48:00:00
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



# # For TSP size 50
# # A5: Basic
# python main.py --variation='A5' --task=tsp50 --decoder=pointer --emb_type=linear --n_glimpses=0 --rnn_layers=1 --n_train=35000

# # B5: Decoder change + Basic # something's wrong
# python main.py --variation='B5' --task=tsp50 --decoder=self --emb_type=linear --n_glimpses=0 --n_train=35000 --num_heads=1 --embedding_dim=128

# # C5: Emb type change + Basic
# python main.py --variation='C5' --task=tsp50 --decoder=pointer --emb_type=enhanced --n_glimpses=0 --rnn_layers=1 --n_train=35000

# # D5: New combination # something's wrong
# python main.py --variation='D5' --task=tsp50 --decoder=self --emb_type=enhanced --n_glimpses=0 --n_train=35000 --num_heads=1

# # E5: Enhanced emb + Glimpse
# python main.py --variation='E5' --task=tsp50 --decoder=pointer --emb_type=enhanced --n_glimpses=3 --rnn_layers=1 --n_train=35000

# # F5: New combination + rnn # something's wrong
# python main.py --variation='F5' --task=tsp50 --decoder=self --emb_type=enhanced --n_glimpses=0 --rnn_layers=2 --n_train=35000 --num_heads=1

# # For TSP size 100
# # A10: Basic
# python main.py --variation='A10' --task=tsp100 --decoder=pointer --emb_type=linear --n_glimpses=0 --rnn_layers=1 --n_train=10000

# # B10: Decoder change + Basic # something's wrong
# python main.py --variation='B10' --task=tsp100 --decoder=self --emb_type=linear --n_glimpses=0 --n_train=10000 --num_heads=1

# # C10: Emb type change + Basic
# python main.py --variation='C10' --task=tsp100 --decoder=pointer --emb_type=enhanced --n_glimpses=0 --rnn_layers=1 --n_train=10000

# # D10: New combination # something's wrong
# python main.py --variation='D10' --task=tsp100 --decoder=self --emb_type=enhanced --n_glimpses=0 --n_train=10000 --num_heads=1

# # E10: Enhanced emb + Glimpse # out of memory
# python main.py --variation='E10' --task=tsp100 --decoder=pointer --emb_type=enhanced --n_glimpses=3 --rnn_layers=1 --n_train=10000

# # F10: New combination + rnn
# python main.py --variation='F10' --task=tsp100 --decoder=self --emb_type=enhanced --n_glimpses=0 --rnn_layers=2 --n_train=10000 --num_heads=1


##### EVAL

## Eval A1
# ## Batch
# python main.py --variation='A1' --task=tsp10 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/tsp10/A1/model/agent_complete.pth --infer_type=batch
## Single
# python main.py --variation='A1' --task=tsp10 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/tsp10/A1/model/agent_complete.pth --infer_type=single

## Eval B1
# ## Batch
# python main.py --variation='B1' --task=tsp10 --is_train=False --decoder=self --emb_type=linear  --num_heads=1 --model_dir=./logs/tsp10/B1/model/agent_complete.pth --infer_type=batch
## Single
# python main.py --variation='B1' --task=tsp10 --is_train=False --decoder=self --emb_type=linear  --num_heads=1 --model_dir=./logs/tsp10/A1/model/agent_complete.pth --infer_type=single


# ## Eval C1
# ## Single
# python main.py --variation='C1' --task=tsp10 --is_train=False  --decoder=pointer --emb_type=enhanced --model_dir=./logs/C1/model/agent_complete.pth --infer_type=single
# ## Batch
# python main.py --variation='C1' --task=tsp10 --is_train=False  --decoder=pointer --emb_type=enhanced --model_dir=./logs/C1/model/agent_complete.pth --infer_type=batch

## Eval A2
## Single
# python main.py --variation='A2' --task=tsp20 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/tsp20/A2/model/agent_complete.pth --infer_type=single
# ## Batch
# python main.py --variation='A2' --task=tsp20 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/tsp20/A2/model/agent_complete.pth --infer_type=batch

## Eval C2
## Single
# python main.py --variation='C2' --task=tsp20 --is_train=False  --decoder=pointer --emb_type=enhanced --model_dir=./logs/tsp20/C2/model/agent_complete.pth --infer_type=single
# Batch
# python main.py --variation='C2' --task=tsp20 --is_train=False  --decoder=pointer --emb_type=enhanced --model_dir=./logs/tsp20/C2/model/agent_complete.pth --infer_type=batch

###########################

## Eval A1
## Single
# python main.py --variation='A1' --task=tsp10 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/A1/model/agent_complete.pth --infer_type=single
# ## Batch
# python main.py --variation='A1' --task=tsp10 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/A1/model/agent_complete.pth --infer_type=batch


echo "[$SHELL] ## Script finished"