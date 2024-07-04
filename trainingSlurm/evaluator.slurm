#!/bin/bash
#SBATCH --job-name=Eval_DSA_1H_10_conv
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

python main.py --variation='DSA_1H_10_conv' --task=tsp10 --n_train=100000 --decoder=self --emb_type=conv  --num_heads=1 --is_train=False --infer_type=batch --model_dir=./logs/DSA_1H_10_conv/model/agent_complete.pth 


##### EVAL
#--is_train=False --infer_type=batch --model_dir=./logs/DPN_10_lin/model/agent_complete.pth 


echo "[$SHELL] ## Script finished"