#!/bin/bash

#SBATCH --job-name=Job_python
#SBATCH --output=resultat.txt
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem=4096
#SBATCH --exclusive

source /etc/profile.d/modules.sh
module load anaconda3
conda activate /usagers4/p117620/anaconda3/envs/rl4e2e

cd ../../../
WORK_DIR=$(pwd)
# echo $WORK_DIR
# pip install --no-index -r requirementscc.txt

cd RL4E2E

# python -m spacy download en_core_web_sm

python train_pdqn.py \
    --episodes=2000 \
    --epsilon_steps=10000 \
    --action=test \
    --model=pptod \
    --version=2.0