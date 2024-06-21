#!/bin/bash
#SBATCH --job-name=Training_A1_min
#SBATCH --output=outputs/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=120:00:00
#SBATCH --partition=gpu-long
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




echo "[$SHELL] ## Script finished"


# Training organized

# A1: Basic
# python main.py --variation='A1_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear 
# called C1 previously
# python main.py --variation='A1_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced  
# python main.py --variation='A1_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 
python main.py --variation='A1_min' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=minimal 

# B1: Decoder change + Basic
# Also test with big epsilon if bad results
# python main.py --variation='B1_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1
# called D1 previously
# python main.py --variation='B1_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1
# python main.py --variation='B1_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1
# python main.py --variation='B1_min' --task=tsp10 --n_train=100000 --decoder=self --emb_type=minimal  --num_heads=1

# D1: Heads alteration - self
# python main.py --variation='D1_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=4
# python main.py --variation='D1_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=4
# python main.py --variation='D1_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=4
# python main.py --variation='D1_min' --task=tsp10 --n_train=100000 --decoder=self --emb_type=minimal  --num_heads=4

# E1: Emb var + Glimpse
# python main.py --variation='E1_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --n_glimpses=3 
# python main.py --variation='E1_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 
# python main.py --variation='E1_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3 
# python main.py --variation='E1_min' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=minimal --n_glimpses=3 

# F1: Rnn layers + self
# python main.py --variation='F1_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --rnn_layers=2 --num_heads=4 
# python main.py --variation='F1_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --rnn_layers=2 --num_heads=4 
# python main.py --variation='F1_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --rnn_layers=2 --num_heads=4 
# python main.py --variation='F1_min' --task=tsp10 --n_train=100000 --decoder=self --emb_type=minimal  --rnn_layers=2 --num_heads=4 

#### TSP20

# A2: Basic
# python main.py --variation='A2_lin' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=linear 
# called C2 previously
# python main.py --variation='A2_en' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced  
# python main.py --variation='A2_en2' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced2 
# python main.py --variation='A2_min' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=minimal 

# B2: Decoder change + Basic
# Also test with big epsilon if bad results
# python main.py --variation='B2_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1
# called D2 previously
# python main.py --variation='B2_en' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1
# python main.py --variation='B2_en2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1
# python main.py --variation='B2_min' --task=tsp20 --n_train=100000 --decoder=self --emb_type=minimal  --num_heads=1

# D2: Heads alteration - self
# python main.py --variation='D2_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=4
# python main.py --variation='D2_en' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=4
# python main.py --variation='D2_en2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=4
# python main.py --variation='D2_min' --task=tsp20 --n_train=100000 --decoder=self --emb_type=minimal  --num_heads=4

# E2: Emb var + Glimpse
# python main.py --variation='E2_lin' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=linear --n_glimpses=3 
# python main.py --variation='E2_en' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 
# python main.py --variation='E2_en2' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3
# python main.py --variation='E2_min' --task=tsp20 --n_train=100000 --decoder=pointer --emb_type=minimal --n_glimpses=3

# F2: Rnn layers + self
# python main.py --variation='F2_lin' --task=tsp20 --n_train=100000 --decoder=self --emb_type=linear --rnn_layers=2 --num_heads=4
# python main.py --variation='F2_en' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced --rnn_layers=2 --num_heads=4
# python main.py --variation='F2_en2' --task=tsp20 --n_train=100000 --decoder=self --emb_type=enhanced2 --rnn_layers=2 --num_heads=4
# python main.py --variation='F2_min' --task=tsp20 --n_train=100000 --decoder=self --emb_type=minimal --rnn_layers=2 --num_heads=4

# _______________________________________________________
# _______________________________________________________

#### TSP50
# A5: Basic
# python main.py --variation='A5_lin' --task=tsp50 --n_train=35000 --decoder=pointer --emb_type=linear 
# python main.py --variation='A5_en' --task=tsp50 --n_train=35000 --decoder=pointer --emb_type=enhanced  
# python main.py --variation='A5_en2' --task=tsp50 --n_train=35000 --decoder=pointer --emb_type=enhanced2 
# python main.py --variation='A5_min' --task=tsp50 --n_train=35000 --decoder=pointer --emb_type=minimal 

# B5: Decoder change + Basic
# Also test with big epsilon if bad results
# python main.py --variation='B5_lin' --task=tsp50 --n_train=35000 --decoder=self --emb_type=linear  --num_heads=1
# python main.py --variation='B5_en' --task=tsp50 --n_train=35000 --decoder=self --emb_type=enhanced  --num_heads=1
# python main.py --variation='B5_en2' --task=tsp50 --n_train=35000 --decoder=self --emb_type=enhanced2  --num_heads=1
# python main.py --variation='B5_min' --task=tsp50 --n_train=35000 --decoder=self --emb_type=minimal  --num_heads=1

# D5: Heads alteration - self
# python main.py --variation='D5_lin' --task=tsp50 --n_train=35000 --decoder=self --emb_type=linear  --num_heads=4
# python main.py --variation='D5_en' --task=tsp50 --n_train=35000 --decoder=self --emb_type=enhanced  --num_heads=4
# python main.py --variation='D5_en2' --task=tsp50 --n_train=35000 --decoder=self --emb_type=enhanced2  --num_heads=4
# python main.py --variation='D5_min' --task=tsp50 --n_train=35000 --decoder=self --emb_type=minimal  --num_heads=4

# E5: Emb var + Glimpse
# python main.py --variation='E5_lin' --task=tsp50 --n_train=35000 --decoder=pointer --emb_type=linear --n_glimpses=3 
# python main.py --variation='E5_en' --task=tsp50 --n_train=35000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 
# python main.py --variation='E5_en2' --task=tsp50 --n_train=35000 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3
# python main.py --variation='E5_min' --task=tsp50 --n_train=35000 --decoder=pointer --emb_type=minimal --n_glimpses=3

# F5: Rnn layers + self
# python main.py --variation='F5_lin' --task=tsp50 --n_train=35000 --decoder=self --emb_type=linear --rnn_layers=2 --num_heads=4
# python main.py --variation='F5_en' --task=tsp50 --n_train=35000 --decoder=self --emb_type=enhanced --rnn_layers=2 --num_heads=4
# python main.py --variation='F5_en2' --task=tsp50 --n_train=35000 --decoder=self --emb_type=enhanced2 --rnn_layers=2 --num_heads=4
# python main.py --variation='F5_min' --task=tsp50 --n_train=35000 --decoder=self --emb_type=minimal --rnn_layers=2 --num_heads=4


#### TSP100

# A10: Basic
# python main.py --variation='A10_lin' --task=tsp100 --n_train=15000 --decoder=pointer --emb_type=linear 
# python main.py --variation='A10_en' --task=tsp100 --n_train=15000 --decoder=pointer --emb_type=enhanced  
# python main.py --variation='A10_en2' --task=tsp100 --n_train=15000 --decoder=pointer --emb_type=enhanced2 
# python main.py --variation='A10_min' --task=tsp100 --n_train=15000 --decoder=pointer --emb_type=minimal 

# B10: Decoder change + Basic
# python main.py --variation='B10_lin' --task=tsp100 --n_train=15000 --decoder=self --emb_type=linear  --num_heads=1
# python main.py --variation='B10_en' --task=tsp100 --n_train=15000 --decoder=self --emb_type=enhanced  --num_heads=1
# python main.py --variation='B10_en2' --task=tsp100 --n_train=15000 --decoder=self --emb_type=enhanced2  --num_heads=1
# python main.py --variation='B10_min' --task=tsp100 --n_train=15000 --decoder=self --emb_type=minimal  --num_heads=1

# D10: Heads alteration - self
# python main.py --variation='D10_lin' --task=tsp100 --n_train=15000 --decoder=self --emb_type=linear  --num_heads=4
# python main.py --variation='D10_en' --task=tsp100 --n_train=15000 --decoder=self --emb_type=enhanced  --num_heads=4
# python main.py --variation='D10_en2' --task=tsp100 --n_train=15000 --decoder=self --emb_type=enhanced2  --num_heads=4
# python main.py --variation='D10_min' --task=tsp100 --n_train=15000 --decoder=self --emb_type=minimal  --num_heads=4

# E10: Emb var + Glimpse
# python main.py --variation='E10_lin' --task=tsp100 --n_train=15000 --decoder=pointer --emb_type=linear --n_glimpses=3 
# python main.py --variation='E10_en' --task=tsp100 --n_train=15000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 
# python main.py --variation='E10_en2' --task=tsp100 --n_train=15000 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3
# python main.py --variation='E10_min' --task=tsp100 --n_train=15000 --decoder=pointer --emb_type=minimal --n_glimpses=3

# F10: Rnn layers + self
# python main.py --variation='F10_lin' --task=tsp100 --n_train=15000 --decoder=self --emb_type=linear --rnn_layers=2 --num_heads=4
# python main.py --variation='F10_en' --task=tsp100 --n_train=15000 --decoder=self --emb_type=enhanced --rnn_layers=2 --num_heads=4
# python main.py --variation='F10_en2' --task=tsp100 --n_train=15000 --decoder=self --emb_type=enhanced2 --rnn_layers=2 --num_heads=4
# python main.py --variation='F10_min' --task=tsp100 --n_train=15000 --decoder=self --emb_type=minimal --rnn_layers=2 --num_heads=4
