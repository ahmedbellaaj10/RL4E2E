#!/bin/bash

cd ../../../
WORK_DIR=$(pwd)
echo $WORK_DIR

cd RL4E2E

python train_pdqn.py \
    --episodes=2000 \
    --epsilon_steps=10000 \
    --action=test \
    --model=pptod \
    --version=2.0