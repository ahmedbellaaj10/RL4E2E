#!/bin/bash

cd ../../../
WORK_DIR=$(pwd)
echo $WORK_DIR

cd RL4E2E

python train_pdqn.py \
    --episodes=10000 \
    --epsilon_steps=1000 \
    --action=train \
    --model=pptod \
    --version=2.1 \
    --num_selected_actions=1