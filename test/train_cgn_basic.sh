#!/bin/bash 
# this is currently broken
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
cd $SGADHOME
CUDA_LAUNCH_BLOCKING=1 python cgn/train_cgn.py --outpath $DIR/_tmp --model_name cgn_test --cfg $DIR/data/basic_cgn_cfg.yaml --target_class 1
