#!/bin/bash
#SBATCH --account=def-foutsekh
#SBATCH --mem-per-cpu=4G
#SBATCH --time=168:00:00


module load python/3.8.10
virtualenv --no-download ENV
source ENV/bin/activate

cd ../../../
WORK_DIR=$(pwd)
echo $WORK_DIR
pip install --no-index -r requirementscc.txt

cd RL4E2E

python -m spacy download en_core_web_sm
python -m gensim.downloader --glove-wiki-gigaword-100

python train_pdqn.py \
    --episodes=2000 \
    --epsilon_steps=10000 \
    --action=test \
    --model=galaxy \
    --version=2.0