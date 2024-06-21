#!/bin/bash
#SBATCH --job-name=Eval_All2s
#SBATCH --output=eval/%x_%j.out
#SBATCH --mail-user="s3516423@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=00:10:00
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

# Eval organized

# python main.py --variation='A2_lin' --task=tsp20 --decoder=pointer --emb_type=linear --is_train=False --model_dir=./logs/A2_lin/model/agent_complete.pth --infer_type=batch
# python main.py --variation='A2_lin' --task=tsp20 --decoder=pointer --emb_type=linear --is_train=False --model_dir=./logs/A2_lin_2/model/agent_complete.pth --infer_type=batch
# python main.py --variation='A2_en' --task=tsp20 --decoder=pointer --emb_type=enhanced --is_train=False --model_dir=./logs/A2_en/model/agent_complete.pth --infer_type=batch
# python main.py --variation='A2_en' --task=tsp20 --decoder=pointer --emb_type=enhanced --is_train=False --model_dir=./logs/A2_en_2/model/agent_complete.pth --infer_type=batch
# python main.py --variation='A2_en2' --task=tsp20 --decoder=pointer --emb_type=enhanced2 --is_train=False --model_dir=./logs/A2_en2/model/agent_complete.pth --infer_type=batch
# python main.py --variation='A2_min' --task=tsp20 --decoder=pointer --emb_type=minimal --is_train=False --model_dir=./logs/A2_min/model/agent_complete.pth --infer_type=batch

# python main.py --variation='B2_lin' --task=tsp20 --decoder=self --emb_type=linear --num_heads=1 --is_train=False --model_dir=./logs/B2_lin/model/agent_complete.pth --infer_type=batch
# python main.py --variation='B2_en' --task=tsp20 --decoder=self --emb_type=enhanced --num_heads=1 --is_train=False --model_dir=./logs/B2_en/model/agent_complete.pth --infer_type=batch
# python main.py --variation='B2_en2' --task=tsp20 --decoder=self --emb_type=enhanced2 --num_heads=1 --is_train=False --model_dir=./logs/B2_en2/model/agent_complete.pth --infer_type=batch
# python main.py --variation='B2_min' --task=tsp20 --decoder=self --emb_type=minimal --num_heads=1 --is_train=False --model_dir=./logs/B2_min/model/agent_complete.pth --infer_type=batch
# python main.py --variation='B2_min' --task=tsp20 --decoder=self --emb_type=minimal --num_heads=1 --is_train=False --model_dir=./logs/B2_min_2/model/agent_complete.pth --infer_type=batch
# python main.py --variation='D2_lin' --task=tsp20 --decoder=self --emb_type=linear --num_heads=4 --is_train=False --model_dir=./logs/D2_lin/model/agent_complete.pth --infer_type=batch
# python main.py --variation='E2_lin' --task=tsp20 --decoder=pointer --emb_type=linear --n_glimpses=3 --is_train=False --model_dir=./logs/E2_lin/model/agent_complete.pth --infer_type=batch
# python main.py --variation='E2_en2' --task=tsp20 --decoder=pointer --emb_type=enhanced2 --n_glimpses=3 --is_train=False --model_dir=./logs/E2_en2/model/agent_complete.pth --infer_type=batch


##### EVAL

## Eval A1
# ## Batch
# python main.py --variation='A1' --task=tsp10 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/tsp10/A1/model/agent_complete.pth --infer_type=batch
## Single
# python main.py --variation='A1' --task=tsp10 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/tsp10/A1/model/agent_complete.pth --infer_type=single


echo "[$SHELL] ## Script finished"