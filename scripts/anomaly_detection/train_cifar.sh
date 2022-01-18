#!/bin/bash 
CLASS=$1
OUTPATH=/home/skvara/work/counterfactual_ad/data/models/anomaly_detection/cifar10/class=$CLASS
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
python train.py --seed 1 --outpath $OUTPATH --cfg configs/cgn_cfg.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/cgn_cfg.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/cgn_cfg.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/cgn_cfg.yaml --target_class $CLASS

python train.py --seed 1 --outpath $OUTPATH --cfg configs/batchsize.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/batchsize.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/batchsize.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/batchsize.yaml --target_class $CLASS

python train.py --seed 1 --outpath $OUTPATH --cfg configs/latent.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/latent.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/latent.yaml --target_class $CLASS
python train.py --seed 1 --outpath $OUTPATH --cfg configs/latent.yaml --target_class $CLASS
