#!/bin/bash 
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
cd $SGADHOME
python mnists/train_cgn.py --outpath $DIR/_tmp --model_name cgn_test --cfg $DIR/data/basic_cgn_cfg.yaml
