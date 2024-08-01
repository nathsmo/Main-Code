#!/bin/bash
#SBATCH --job-name=Eval-DPN_100_ep3_en2
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


# D# DSA_2H_100: Self-Attention Decoder (1 head) - 0 Epsilon
# python main.py --variation='DSA_1H_100_ep0_conv' --task=tsp100 --decoder=self --emb_type=conv --num_heads=1 --epsilon=0 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-17-48/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep0_lin' --task=tsp100 --decoder=self --emb_type=linear --num_heads=1 --epsilon=0 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-25-55/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep0_en' --task=tsp100 --decoder=self --emb_type=enhanced --num_heads=1 --epsilon=0 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-28-58/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep0_en2' --task=tsp100 --decoder=self --emb_type=enhanced2 --num_heads=1 --epsilon=0 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-29-08/model/best_model.pth

# DSA_2H_100: Self-Attention Decoder (1 head) - 0.1 Epsilon
# python main.py --variation='DSA_1H_100_ep1_conv' --task=tsp100 --decoder=self --emb_type=conv --num_heads=1 --epsilon=0.1 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-30-38/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep1_lin' --task=tsp100 --decoder=self --emb_type=linear --num_heads=1 --epsilon=0.1 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-32-38/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep1_en' --task=tsp100 --decoder=self --emb_type=enhanced --num_heads=1 --epsilon=0.1 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-33-11/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep1_en2' --task=tsp100 --decoder=self --emb_type=enhanced2 --num_heads=1 --epsilon=0.1 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-33-34/model/best_model.pth

# DSA_2H_100: Self-Attention Decoder (1 head) - 0.3 Epsilon
# python main.py --variation='DSA_1H_100_ep3_conv' --task=tsp100 --decoder=self --emb_type=conv --num_heads=1 --epsilon=0.3 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-35-40/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep3_lin' --task=tsp100 --decoder=self --emb_type=linear --num_heads=1 --epsilon=0.3 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-40-21/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep3_en' --task=tsp100 --decoder=self --emb_type=enhanced --num_heads=1 --epsilon=0.3 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-41-51/model/best_model.pth
# python main.py --variation='DSA_1H_100_ep3_en2' --task=tsp100 --decoder=self --emb_type=enhanced2 --num_heads=1 --epsilon=0.3 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-54-35/model/best_model.pth

# DPN_100: Basic Pointer Network - Epsilon 0 
# python main.py --variation='DPN_100_ep0_conv' --task=tsp100 --decoder=pointer --emb_type=conv --epsilon=0 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-57-00/model/best_model.pth
# python main.py --variation='DPN_100_ep0_lin' --task=tsp100 --decoder=pointer --emb_type=linear --epsilon=0 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-57-11/model/best_model.pth
# python main.py --variation='DPN_100_ep0_en' --task=tsp100 --decoder=pointer --emb_type=enhanced --epsilon=0 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_00-59-11/model/best_model.pth
# python main.py --variation='DPN_100_ep0_en2' --task=tsp100 --decoder=pointer --emb_type=enhanced2 --epsilon=0 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_01-00-49/model/best_model.pth

# DPN_100: Basic Pointer Network - Epsilon 0.1
# python main.py --variation='DPN_100_ep1_conv' --task=tsp100 --decoder=pointer --emb_type=conv --epsilon=0.1 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_01-03-05/model/best_model.pth
# python main.py --variation='DPN_100_ep1_lin' --task=tsp100 --decoder=pointer --emb_type=linear --epsilon=0.1 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_01-05-14/model/best_model.pth
# python main.py --variation='DPN_100_ep1_en' --task=tsp100 --decoder=pointer --emb_type=enhanced --epsilon=0.1 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_01-11-11/model/best_model.pth
# python main.py --variation='DPN_100_ep1_en2' --task=tsp100 --decoder=pointer --emb_type=enhanced2 --epsilon=0.1 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_01-13-12/model/best_model.pth

# DPN_100: Basic Pointer Network - Epsilon 0.3
# python main.py --variation='DPN_100_ep3_conv' --task=tsp100 --decoder=pointer --emb_type=conv --epsilon=0.3 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_01-33-23/model/best_model.pth
# python main.py --variation='DPN_100_ep3_lin' --task=tsp100 --decoder=pointer --emb_type=linear --epsilon=0.3 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_01-50-03/model/best_model.pth
# python main.py --variation='DPN_100_ep3_en' --task=tsp100 --decoder=pointer --emb_type=enhanced --epsilon=0.3 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_02-33-43/model/best_model.pth
python main.py --variation='DPN_100_ep3_en2' --task=tsp100 --decoder=pointer --emb_type=enhanced2 --epsilon=0.3 --batch_size=128 --is_train=False --infer_type=batch --model_dir=./logs/tsp100-2024-08-01_02-40-13/model/best_model.pth


echo "[$SHELL] ## Script finished"
