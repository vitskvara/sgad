#!/bin/bash 
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
cd $SGADHOME
python cgn/train_cgn.py --cfg cgn/experiments/cgn_wildlife_MNIST/cfg.yaml
cd $DIR