#!/bin/bash

cd ../../../
WORK_DIR=$(pwd)
echo $WORK_DIR

cd RL4E2E

python train_pdqn.py \
    --action=test \
    --model=galaxy \
    --version=2.0
    --num_selected_actions=6