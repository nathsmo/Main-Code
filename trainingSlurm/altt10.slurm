#!/bin/bash
#SBATCH --job-name=Training_DSA_10_2H_1RNN_conv
#SBATCH --output=outputs/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-long
#SBATCH --ntasks=4

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

# DPN_10: Pointer Network - Decoder - 1 RNN
# python main.py --variation='DPN_10_3G_conv_1RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --rnn_layers=1 --batch_size=64 --epsilon=0 --n_glimpses=3
# python main.py --variation='DPN_10_3G_lin_1RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --rnn_layers=1 --batch_size=64 --epsilon=0 --n_glimpses=3
# python main.py --variation='DPN_10_3G_en_1RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --rnn_layers=1 --batch_size=64 --epsilon=0 --n_glimpses=3
# python main.py --variation='DPN_10_3G_en2_1RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --rnn_layers=1 --batch_size=64 --epsilon=0 --n_glimpses=3

# DPN_10: Pointer Network - Decoder - 2 RNN
# python main.py --variation='DPN_10_3G_conv_2RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --rnn_layers=2 --batch_size=64 --epsilon=0 --n_glimpses=3
# python main.py --variation='DPN_10_3G_lin_2RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --rnn_layers=2 --batch_size=64 --epsilon=0 --n_glimpses=3
# python main.py --variation='DPN_10_3G_en_2RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --rnn_layers=2 --batch_size=64 --epsilon=0 --n_glimpses=3
# python main.py --variation='DPN_10_3G_en2_2RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --rnn_layers=2 --batch_size=64 --epsilon=0 --n_glimpses=3

# DSA_2H_10: Self-Attention Decoder (2 head) + 1RNN
# python main.py --variation='DSA_10_2H_1RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=2 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_2H_1RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=2 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_2H_1RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=2 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_2H_1RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=2 --rnn_layers=1 --batch_size=128 --epsilon=0.1

# DSA_2H_10: Self-Attention Decoder (2 head) + 2RNN
# python main.py --variation='DSA_10_2H_2RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=2 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_2H_2RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=2 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_2H_2RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=2 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_2H_2RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=2 --rnn_layers=2 --batch_size=128 --epsilon=0.1

# DSA_4H_10: Self-Attention Decoder (4 head) + 1RNN
# python main.py --variation='DSA_10_4H_1RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=4 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_4H_1RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=4 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_4H_1RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=4 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_4H_1RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=4 --rnn_layers=1 --batch_size=128 --epsilon=0.1

# DSA_4H_10: Self-Attention Decoder (4 head) + 2RNN
# python main.py --variation='DSA_10_4H_2RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=4 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_4H_2RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=4 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_4H_2RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=4 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_4H_2RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=4 --rnn_layers=2 --batch_size=128 --epsilon=0.1


# ---------------
# DPN_3G_10: Pointer Network - Decoder + 3 Glimpses + 2 RNN
# python main.py --variation='DPN_3G_2RNN_10_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --n_glimpses=3 --rnn_layers=2
# python main.py --variation='DPN_3G_2RNN_10_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --n_glimpses=3 --rnn_layers=2
# python main.py --variation='DPN_3G_2RNN_10_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --n_glimpses=3 --rnn_layers=2
# python main.py --variation='DPN_3G_2RNN_10_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3 --rnn_layers=2


###### 

# DSA_2H_10: Self-Attention Decoder (1 head) + 1RNN
# python main.py --variation='DSA_10_1H_1RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_1RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_1RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_1RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --rnn_layers=1 --batch_size=128 --epsilon=0.1

# DSA_2H_10: Self-Attention Decoder (1 head) + 2RNN
# python main.py --variation='DSA_10_1H_2RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_2RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_2RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_2RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --rnn_layers=2 --batch_size=128 --epsilon=0.1

# --------------------
# DSA_2H_10: Self-Attention Decoder (2 head)
# python main.py --variation='DSA_2H_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=2
# python main.py --variation='DSA_2H_10_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=2
# python main.py --variation='DSA_2H_10_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=2
# python main.py --variation='DSA_2H_10_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=2

# DSA_2H_10: Self-Attention Decoder (1 head) + 2 RNN
# python main.py --variation='DSA_1H_10_2RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --rnn_layers=2
# python main.py --variation='DSA_1H_10_2RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --rnn_layers=2
# python main.py --variation='DSA_1H_10_2RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --rnn_layers=2
# python main.py --variation='DSA_1H_10_2RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --rnn_layers=2

# DSA_2H_10: Self-Attention Decoder (2 head) + 2 RNN
# python main.py --variation='DSA_2H_10_2RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=2 --rnn_layers=2
# python main.py --variation='DSA_2H_10_2RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=2 --rnn_layers=2
# python main.py --variation='DSA_2H_10_2RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=2 --rnn_layers=2
# python main.py --variation='DSA_2H_10_2RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=2 --rnn_layers=2




####

# DSA_10: Self-Attention - Decoder - batch_size=64
# python main.py --variation='DSA_10_bs64_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv --batch_size=64 --num_heads=1
# python main.py --variation='DSA_10_bs64_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear --batch_size=64 --num_heads=1
# python main.py --variation='DSA_10_bs64_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced --batch_size=64 --num_heads=1
# python main.py --variation='DSA_10_bs64_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2 --batch_size=64 --num_heads=1

# DSA_10: Self-Attention - Decoder - batch_size=128
# python main.py --variation='DSA_10_bs128_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv --batch_size=128 --num_heads=1
# python main.py --variation='DSA_10_bs128_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear --batch_size=128 --num_heads=1
# python main.py --variation='DSA_10_bs128_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced --batch_size=128 --num_heads=1
# python main.py --variation='DSA_10_bs128_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2 --batch_size=128 --num_heads=1

# DSA_10: Self-Attention - Decoder - batch_size=256
# python main.py --variation='DSA_10_bs256_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv --batch_size=256 --num_heads=1
# python main.py --variation='DSA_10_bs256_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear --batch_size=256 --num_heads=1
# python main.py --variation='DSA_10_bs256_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced --batch_size=256 --num_heads=1
# python main.py --variation='DSA_10_bs256_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2 --batch_size=256 --num_heads=1

# DPN_10: Basic Pointer Network - Decoder - batch_size=256
# python main.py --variation='DPN_10_bs256_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --batch_size=256
# python main.py --variation='DPN_10_bs256_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --batch_size=256
# python main.py --variation='DPN_10_bs256_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --batch_size=256 
# python main.py --variation='DPN_10_bs256_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --batch_size=256

# DPN_10: Basic Pointer Network - Decoder - batch_size=128
# python main.py --variation='DPN_10_bs128_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --batch_size=128
# python main.py --variation='DPN_10_bs128_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --batch_size=128
# python main.py --variation='DPN_10_bs128_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --batch_size=128 
# python main.py --variation='DPN_10_bs128_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --batch_size=128

# DPN_10: Basic Pointer Network - Decoder - batch_size=64
# python main.py --variation='DPN_10_bs64_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --batch_size=64
# python main.py --variation='DPN_10_bs64_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --batch_size=64
# python main.py --variation='DPN_10_bs64_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --batch_size=64 
# python main.py --variation='DPN_10_bs64_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --batch_size=64

### ---------------------------------------------------------------
# DSA_2H_10: Self-Attention Decoder (1 head) - 0 Epsilon
# python main.py --variation='DSA_1H_10_ep0_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --epsilon=0 --batch_size=128
# python main.py --variation='DSA_1H_10_ep0_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --epsilon=0 --batch_size=128
# python main.py --variation='DSA_1H_10_ep0_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --epsilon=0 --batch_size=128
# python main.py --variation='DSA_1H_10_ep0_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --epsilon=0 --batch_size=128

# DSA_2H_10: Self-Attention Decoder (1 head) - 0.1 Epsilon
# python main.py --variation='DSA_1H_10_ep1_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --epsilon=0.1 --batch_size=128
# python main.py --variation='DSA_1H_10_ep1_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --epsilon=0.1 --batch_size=128
# python main.py --variation='DSA_1H_10_ep1_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --epsilon=0.1 --batch_size=128
# python main.py --variation='DSA_1H_10_ep1_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --epsilon=0.1 --batch_size=128

# DSA_2H_10: Self-Attention Decoder (1 head) - 0.3 Epsilon
# python main.py --variation='DSA_1H_10_ep3_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --epsilon=0.3 --batch_size=128
# python main.py --variation='DSA_1H_10_ep3_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --epsilon=0.3 --batch_size=128
# python main.py --variation='DSA_1H_10_ep3_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --epsilon=0.3 --batch_size=128
# python main.py --variation='DSA_1H_10_ep3_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --epsilon=0.3 --batch_size=128

# DPN_10: Basic Pointer Network - Epsilon 0 
# python main.py --variation='DPN_10_ep0_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --epsilon=0 --batch_size=64
# python main.py --variation='DPN_10_ep0_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --epsilon=0 --batch_size=64
# python main.py --variation='DPN_10_ep0_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --epsilon=0 --batch_size=64
# python main.py --variation='DPN_10_ep0_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --epsilon=0 --batch_size=64

# DPN_10: Basic Pointer Network - Epsilon 0.1
# python main.py --variation='DPN_10_ep1_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --epsilon=0.1 --batch_size=64
# python main.py --variation='DPN_10_ep1_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --epsilon=0.1 --batch_size=64
# python main.py --variation='DPN_10_ep1_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --epsilon=0.1 --batch_size=64
# python main.py --variation='DPN_10_ep1_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --epsilon=0.1 --batch_size=64

# DPN_10: Basic Pointer Network - Epsilon 0.3
# python main.py --variation='DPN_10_ep3_conv' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --epsilon=0.3 --batch_size=64
# python main.py --variation='DPN_10_ep3_lin' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --epsilon=0.3 --batch_size=64
# python main.py --variation='DPN_10_ep3_en' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --epsilon=0.3 --batch_size=64
# python main.py --variation='DPN_10_ep3_en2' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --epsilon=0.3 --batch_size=64

# DPN_10: Pointer Network - Decoder - 1 RNN
# python main.py --variation='DPN_10_conv_1RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --rnn_layers=1 --batch_size=64 --epsilon=0
# python main.py --variation='DPN_10_lin_1RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --rnn_layers=1 --batch_size=64 --epsilon=0
# python main.py --variation='DPN_10_en_1RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --rnn_layers=1 --batch_size=64 --epsilon=0
# python main.py --variation='DPN_10_en2_1RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --rnn_layers=1 --batch_size=64 --epsilon=0

# DPN_10: Pointer Network - Decoder - 2 RNN
# python main.py --variation='DPN_10_conv_2RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=conv --rnn_layers=2 --batch_size=64 --epsilon=0
# python main.py --variation='DPN_10_lin_2RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=linear --rnn_layers=2 --batch_size=64 --epsilon=0
# python main.py --variation='DPN_10_en_2RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced --rnn_layers=2 --batch_size=64 --epsilon=0
# python main.py --variation='DPN_10_en2_2RNN' --task=tsp10 --n_train=100000 --decoder=pointer --emb_type=enhanced2 --rnn_layers=2 --batch_size=64 --epsilon=0

# DSA_2H_10: Self-Attention Decoder (1 head) + 1RNN
# python main.py --variation='DSA_10_1H_1RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_1RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_1RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --rnn_layers=1 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_1RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --rnn_layers=1 --batch_size=128 --epsilon=0.1

# DSA_2H_10: Self-Attention Decoder (1 head) + 2RNN
# python main.py --variation='DSA_10_1H_2RNN_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_2RNN_lin' --task=tsp10 --n_train=100000 --decoder=self --emb_type=linear  --num_heads=1 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_2RNN_en' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced  --num_heads=1 --rnn_layers=2 --batch_size=128 --epsilon=0.1
# python main.py --variation='DSA_10_1H_2RNN_en2' --task=tsp10 --n_train=100000 --decoder=self --emb_type=enhanced2  --num_heads=1 --rnn_layers=2 --batch_size=128 --epsilon=0.1


# python main.py --variation='DSA_10_bs512_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv --batch_size=512 --num_heads=1


echo "[$SHELL] ## Script finished"

