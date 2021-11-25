#!/bin/bash 
PWD=$(pwd)
CGNHOME=/home/skvara/work/counterfactual_ad/counterfactual_generative_networks
cd $CGNHOME
python mnists/train_cgn.py --cfg mnists/experiments/cgn_wildlife_MNIST/cfg.yaml
cd $PWD