#!/bin/bash

cd ../../../
WORK_DIR=$(pwd)
echo $WORK_DIR
myvirtualenv/bin/activate
cd RL4E2E

python train_pdqn.py \
    --episodes=3 \
    --epsilon_steps=2000 \
    --action=train \
    --model=galaxy \
    --version=2.0
    