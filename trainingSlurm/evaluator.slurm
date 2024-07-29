#!/bin/bash
#SBATCH --job-name=Eval_transfer
#SBATCH --output=eval/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-short
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

# Eval organized
##############################
# DSA_4H_2RNN_50
# python main.py --variation='DSA_4H_2RNN_50_conv' --task=tsp50 --n_train=75000 --decoder=self --emb_type=conv --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_50/DSA_4H_2RNN_50_conv/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_50_en' --task=tsp50 --n_train=75000 --decoder=self --emb_type=enhanced --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_50/DSA_4H_2RNN_50_en/model/agent_complete.pth 

##############################
# DSA_4H_2RNN_20
# python main.py --variation='DSA_4H_2RNN_20_conv' --task=tsp20 --n_train=100000 --decoder=self --emb_type=conv --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_20/DSA_4H_2RNN_20_conv/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_20_conv' --task=tsp20 --n_train=100000 --decoder=self --emb_type=conv --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_20/DSA_4H_2RNN_20_conv_2/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_20_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_20/DSA_4H_2RNN_20_lin/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_20_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_20/DSA_4H_2RNN_20_lin_2/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_20_en' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_20/DSA_4H_2RNN_20_en/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_20_en2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced2 --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_20/DSA_4H_2RNN_20_en2/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_20_en2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced2 --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_20/DSA_4H_2RNN_20_en2_2/model/agent_complete.pth 

##############################
# DSA_4H_2RNN_10
# python main.py --variation='DSA_4H_2RNN_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_10/DSA_4H_2RNN_10_conv/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_10_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_10/DSA_4H_2RNN_10_lin/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_10_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_10/DSA_4H_2RNN_10_lin_2/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_10_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_10/DSA_4H_2RNN_10_en/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_10_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_10/DSA_4H_2RNN_10_en2/model/agent_complete.pth 
# python main.py --variation='DSA_4H_2RNN_10_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --rnn_layers=2 --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H_2RNN/DSA_4H_2RNN_10/DSA_4H_2RNN_10_en2_2/model/agent_complete.pth 

##############################
# DSA_4H_50
# python main.py --variation='DSA_4H_50_en' --task=tsp50 --n_train=75000 --decoder=self --emb_type=enhanced  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_50/DSA_4H_50_en/model/agent_complete.pth 

##############################
# DSA_4H_20
# python main.py --variation='DSA_4H_20_conv' --task=tsp20 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_20/DSA_4H_20_conv/model/agent_complete.pth 
# python main.py --variation='DSA_4H_20_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_20/DSA_4H_20_lin/model/agent_complete.pth 
# python main.py --variation='DSA_4H_20_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_20/DSA_4H_20_lin_2/model/agent_complete.pth 
# python main.py --variation='DSA_4H_20_en' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_20/DSA_4H_20_en/model/agent_complete.pth 
# python main.py --variation='DSA_4H_20_en2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_20/DSA_4H_20_en2/model/agent_complete.pth 
# python main.py --variation='DSA_4H_20_en2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_20/DSA_4H_20_en2_2/model/agent_complete.pth 

##############################
# DSA_4H_10
# python main.py --variation='DSA_4H_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_10/DSA_4H_10_conv/model/agent_complete.pth 
# python main.py --variation='DSA_4H_10_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_10/DSA_4H_10_lin/model/agent_complete.pth 
# python main.py --variation='DSA_4H_10_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_10/DSA_4H_10_en/model/agent_complete.pth
# python main.py --variation='DSA_4H_10_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_10/DSA_4H_10_en_2/model/agent_complete.pth
# python main.py --variation='DSA_4H_10_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=4 --is_train=False --infer_type=batch --model_dir=./logs/DSA_4H/DSA_4H_10/DSA_4H_10_en2/model/agent_complete.pth 

##############################
# DSA_1H_50
# python main.py --variation='DSA_1H_50_conv' --task=tsp50 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_conv_3/model/agent_complete.pth 
# python main.py --variation='DSA_1H_50_lin' --task=tsp50 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_lin_2/model/agent_complete.pth 

##############################
# DSA_1H_20

# python main.py --variation='DSA_1H_20_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_lin/model/agent_complete.pth 

# python main.py --variation='DSA_1H_20_conv' --task=tsp20 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_20/DSA_1H_20_conv/model/agent_complete.pth 
# python main.py --variation='DSA_1H_20_conv' --task=tsp20 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_20/DSA_1H_20_conv_2/model/agent_complete.pth 
# python main.py --variation='DSA_1H_20_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_20/DSA_1H_20_lin/model/agent_complete.pth 
# python main.py --variation='DSA_1H_20_en' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_20/DSA_1H_20_en/model/agent_complete.pth 
# python main.py --variation='DSA_1H_20_en2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_20/DSA_1H_20_en2/model/agent_complete.pth 

##############################
# DSA_1H_10
# python main.py --variation='DSA_1H_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_conv/model/agent_complete.pth 
# python main.py --variation='DSA_1H_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_conv_2/model/agent_complete.pth 
# python main.py --variation='DSA_1H_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_conv_3/model/agent_complete.pth 
# python main.py --variation='DSA_1H_10_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_lin/model/agent_complete.pth 
# python main.py --variation='DSA_1H_10_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_lin_2/model/agent_complete.pth 
# python main.py --variation='DSA_1H_10_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_en/model/agent_complete.pth 
# python main.py --variation='DSA_1H_10_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_en2/model/agent_complete.pth 
# python main.py --variation='DSA_1H_10_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H/DSA_1H_10/DSA_1H_10_en2_2/model/agent_complete.pth 

##############################
# DPN_3G_20

# python main.py --variation='DPN_3G_20_conv' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=conv --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_20/DPN_3G_20_conv/model/agent_complete.pth 
# python main.py --variation='DPN_3G_20_lin' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=linear --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_20/DPN_3G_20_lin/model/agent_complete.pth 
# python main.py --variation='DPN_3G_20_lin' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=linear --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_20/DPN_3G_20_lin_2/model/agent_complete.pth 
# python main.py --variation='DPN_3G_20_en2' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_20/DPN_3G_20_en2/model/agent_complete.pth 

##############################
# DPN_3G_10
# python main.py --variation='DPN_3G_10_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_10/DPN_3G_10_conv/model/agent_complete.pth 
# python main.py --variation='DPN_3G_10_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_10/DPN_3G_10_lin/model/agent_complete.pth 
# python main.py --variation='DPN_3G_10_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_10/DPN_3G_10_lin_2/model/agent_complete.pth 
# python main.py --variation='DPN_3G_10_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_10/DPN_3G_10_lin_3/model/agent_complete.pth 
# python main.py --variation='DPN_3G_10_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_10/DPN_3G_10_en/model/agent_complete.pth 
# python main.py --variation='DPN_3G_10_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_10/DPN_3G_10_en_2/model/agent_complete.pth 
# python main.py --variation='DPN_3G_10_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_10/DPN_3G_10_en_3/model/agent_complete.pth 
# python main.py --variation='DPN_3G_10_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3 --is_train=False --infer_type=batch --model_dir=./logs/DPN_3G/DPN_3G_10/DPN_3G_10_en2/model/agent_complete.pth 

##############################
# DPN_50
# python main.py --variation='DPN_50_conv' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_50/DPN_50_conv/model/agent_complete.pth 
# python main.py --variation='DPN_50_conv' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_50/DPN_50_conv_2/model/agent_complete.pth 
# python main.py --variation='DPN_50_conv' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_50/DPN_50_conv_3/model/agent_complete.pth 
# python main.py --variation='DPN_50_en' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=enhanced --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_50/DPN_50_en/model/agent_complete.pth 

##############################
# DPN_20
# python main.py --variation='DPN_20_conv' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_20/DPN_20_conv/model/agent_complete.pth 
# python main.py --variation='DPN_20_conv' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_20/DPN_20_conv_2/model/agent_complete.pth 
# python main.py --variation='DPN_20_lin' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=linear --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_20/DPN_20_lin/model/agent_complete.pth 
# python main.py --variation='DPN_20_lin' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=linear --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_20/DPN_20_lin_2/model/agent_complete.pth 
# python main.py --variation='DPN_20_lin' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=linear --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_20/DPN_20_lin_3/model/agent_complete.pth 
# python main.py --variation='DPN_20_en' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_20/DPN_20_en/model/agent_complete.pth 
# python main.py --variation='DPN_20_en2' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_20/DPN_20_en2/model/agent_complete.pth 

##############################
# DPN_10
# python main.py --variation='DPN_10_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_conv/model/agent_complete.pth 
# python main.py --variation='DPN_10_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_conv_2/model/agent_complete.pth 
# python main.py --variation='DPN_10_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_conv_3/model/agent_complete.pth 
# python main.py --variation='DPN_10_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_lin/model/agent_complete.pth 
# python main.py --variation='DPN_10_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_lin_2/model/agent_complete.pth 
# python main.py --variation='DPN_10_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_lin_3/model/agent_complete.pth 
# python main.py --variation='DPN_10_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_lin_4/model/agent_complete.pth 
# python main.py --variation='DPN_10_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_en/model/agent_complete.pth 
# python main.py --variation='DPN_10_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_en_2/model/agent_complete.pth 
# python main.py --variation='DPN_10_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_en_3/model/agent_complete.pth 
# python main.py --variation='DPN_10_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_en2/model/agent_complete.pth 
# python main.py --variation='DPN_10_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_10/DPN_10_en2_2/model/agent_complete.pth 

##############################
##############################

# python main.py --variation='DPN_50_conv' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_50/DPN_50_conv/model/agent_complete.pth 
# python main.py --variation='DPN_50_conv' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_50/DPN_50_conv_2/model/agent_complete.pth 
# python main.py --variation='DPN_50_conv' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_50/DPN_50_conv_3/model/agent_complete.pth 
# python main.py --variation='DPN_50_en' --task=tsp50 --n_train=75000 --decoder=pointer --emb_type=enhanced --is_train=False --infer_type=batch --model_dir=./logs/DPN/DPN_50/DPN_50_en/model/agent_complete.pth 

# python main.py --variation='tsp10' --task=tsp10 --decoder=pointer --emb_type=enhanced2 --is_train=False --infer_type=batch --n_glimpses=3 --model_dir=./logs/_DPN_3G_20_en/model/agent_complete.pth 
# python main.py --variation='tsp50' --task=tsp50 --decoder=pointer --emb_type=enhanced2 --is_train=False --infer_type=batch --n_glimpses=3 --model_dir=./logs/_DPN_3G_20_en/model/agent_complete.pth 

python main.py --variation='tsp10' --task=tsp10 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/_DSA_1H_50_conv/model/agent_complete.pth 
# python main.py --variation='tsp20' --task=tsp50 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/_DSA_1H_50_conv/model/agent_complete.pth 

##### EVAL
#--is_train=False --infer_type=batch --model_dir=./logs/DPN_10_lin/model/agent_complete.pth 


echo "[$SHELL] ## Script finished"
