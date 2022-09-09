#!/bin/bash

cd ../../../
WORK_DIR=$(pwd)
echo $WORK_DIR

cd RL4E2E

python train_pdqn.py \
    --seed=1 \
    --episodes=5000 \
    --epsilon_steps=1000 \
    --action=train \
    --model=galaxy \
    --version=2.1 \
    --num_selected_actions=1