#!/bin/bash
#SBATCH --job-name=Eval_DPN_10_conv
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

python main.py --variation='DPN_10_conv' --task=tsp10 --decoder=pointer --emb_type=conv --is_train=False --infer_type=batch --model_dir=./logs/DPN_10_lin/model/agent_complete.pth 

# python main.py --variation='A2_lin' --task=tsp20 --decoder=pointer --emb_type=linear --is_train=False --model_dir=./logs/A2_lin/model/agent_complete.pth --infer_type=batch


##### EVAL

## Eval A1
# ## Batch
# python main.py --variation='A1' --task=tsp10 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/tsp10/A1/model/agent_complete.pth --infer_type=batch
## Single
# python main.py --variation='A1' --task=tsp10 --is_train=False --decoder=pointer --emb_type=linear --model_dir=./logs/tsp10/A1/model/agent_complete.pth --infer_type=single


echo "[$SHELL] ## Script finished"