#!/bin/bash 
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
cd $SGADHOME
python cgn/train_cgn_nonmnist.py --outpath $DIR/_tmp --model_name cgn_test_cifar --cfg $DIR/data/cifar_cgn_cfg.yaml
