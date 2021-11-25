#!/bin/bash 
PWD=$(pwd)
CGNHOME=/home/skvara/work/counterfactual_ad/counterfactual_generative_networks
DATASET=cgn_wildlife_MNIST
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$DATASET/default
cd $CGNHOME
python mnists/generate_data.py \
	--weight_path mnists/experiments/$DATASET/weights/ckp.pth \
	--dataset wildlife_MNIST --no_cfs 10 --dataset_size 100000 --outpath=$OUTPATH
cd $PWD
