#!/bin/bash 
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad
cd $SGADHOME
python mnists/train_cgn.py --cfg mnists/experiments/cgn_wildlife_MNIST/cfg.yaml
cd $DIR