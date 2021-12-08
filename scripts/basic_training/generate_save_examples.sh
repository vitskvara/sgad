#!/bin/bash 
# run via compare_generated.sh
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
WEIGHTPATH=$1
SAVEDIR=$2
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$SAVEDIR
cd $SGADHOME
python cgn/generate_data.py \
	--weight_path $WEIGHTPATH \
	--dataset cifar10 --no_cfs 10 --dataset_size 100000 --outpath=$OUTPATH
cd $DIR
