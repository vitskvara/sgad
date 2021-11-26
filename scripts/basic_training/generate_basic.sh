#!/bin/bash 
DIR=$(pwd)
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
DATASET=wildlife_MNIST
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$DATASET/default
cd $SGADHOME
python cgn/generate_data.py \
	--weight_path cgn/experiments/cgn_$DATASET/weights/ckp.pth \
	--dataset $DATASET --no_cfs 10 --dataset_size 100000 --outpath=$OUTPATH
cd $DIR
python save_generated_images.py $OUTPATH/$DATASET_counterfactual.pth