#!/bin/bash 
# run via compare_generated.sh
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad
WEIGHTPATH=$1
DATASET=$2
SAVEDIR=$3
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$DATASET/$SAVEDIR
cd $SGADHOME
python mnists/generate_data.py \
	--weight_path $WEIGHTPATH \
	--dataset $DATASET --no_cfs 10 --dataset_size 100000 --outpath=$OUTPATH
cd $DIR
python save_generated_images.py $OUTPATH/${DATASET}_counterfactual.pth