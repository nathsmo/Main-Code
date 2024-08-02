#!/bin/bash
#SBATCH --job-name=Eval-DPN_50_en2_2RNN
#SBATCH --output=eval/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=00:30:00
#SBATCH --partition=mem
#SBATCH --ntasks=1
#gpu-short
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
# --is_train=False --infer_type=batch --model_dir=./


# DPN_50: Pointer Network - Decoder - 1 RNN
# python main.py --variation='DPN_50_conv_1RNN' --task=tsp50 --decoder=pointer --emb_type=conv --rnn_layers=1 --batch_size=64 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./logs/tsp50-2024-08-01_12-36-16/model/best_model.pth
# python main.py --variation='DPN_50_lin_1RNN' --task=tsp50 --decoder=pointer --emb_type=linear --rnn_layers=1 --batch_size=64 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./logs/tsp50-2024-08-01_12-36-27/model/best_model.pth
# python main.py --variation='DPN_50_en_1RNN' --task=tsp50 --decoder=pointer --emb_type=enhanced --rnn_layers=1 --batch_size=64 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./logs/tsp50-2024-08-01_12-36-34/model/best_model.pth
# python main.py --variation='DPN_50_en2_1RNN' --task=tsp50 --decoder=pointer --emb_type=enhanced2 --rnn_layers=1 --batch_size=64 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./logs/tsp50-2024-08-01_12-36-39/model/best_model.pth

# DPN_50: Pointer Network - Decoder - 2 RNN
# python main.py --variation='DPN_50_conv_2RNN' --task=tsp50 --decoder=pointer --emb_type=conv --rnn_layers=2 --batch_size=64 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./logs/tsp50-2024-08-01_12-36-46/model/best_model.pth
# python main.py --variation='DPN_50_lin_2RNN' --task=tsp50 --decoder=pointer --emb_type=linear --rnn_layers=2 --batch_size=64 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./logs/tsp50-2024-08-01_12-36-56/model/best_model.pth
# python main.py --variation='DPN_50_en_2RNN' --task=tsp50 --decoder=pointer --emb_type=enhanced --rnn_layers=2 --batch_size=64 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./logs/tsp50-2024-08-01_12-37-02/model/best_model.pth
# python main.py --variation='DPN_50_en2_2RNN' --task=tsp50 --decoder=pointer --emb_type=enhanced2 --rnn_layers=2 --batch_size=64 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./logs/tsp50-2024-08-01_12-37-09/model/best_model.pth



# DSA_4H_100: Self-Attention Decoder (4 head) + 2RNN
# python main.py --variation='DSA_100_4H_2RNN_conv' --task=tsp100 --decoder=self --emb_type=conv --num_heads=4 --rnn_layers=2 --batch_size=128 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./
# python main.py --variation='DSA_100_4H_2RNN_lin' --task=tsp100 --decoder=self --emb_type=linear --num_heads=4 --rnn_layers=2 --batch_size=128 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./
# python main.py --variation='DSA_100_4H_2RNN_en' --task=tsp100 --decoder=self --emb_type=enhanced --num_heads=4 --rnn_layers=2 --batch_size=128 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./
# python main.py --variation='DSA_100_4H_2RNN_en2' --task=tsp100 --decoder=self --emb_type=enhanced2 --num_heads=4 --rnn_layers=2 --batch_size=128 --epsilon=0 --is_train=False --infer_type=batch --model_dir=./


echo "[$SHELL] ## Script finished"
