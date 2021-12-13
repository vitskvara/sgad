#!/bin/bash 
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
cd $SGADHOME
python cgn/train_cgn_nonmnist.py --outpath $DIR/_tmp --cfg $DIR/data/wmnist_cgn_cfg.yaml --seed 1  --target_class 1
