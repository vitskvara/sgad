#!/bin/bash 
CLASS=1
OUTPATH=/home/skvara/work/counterfactual_ad/data/models/cgn_cifar_oneclass/class_$CLASS
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/cgn_cfg.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/cgn_cfg.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/cgn_cfg.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/cgn_cfg.yaml --target_class $CLASS

python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/batchsize.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/batchsize.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/batchsize.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/batchsize.yaml --target_class $CLASS

python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/latent.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/latent.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/latent.yaml --target_class $CLASS
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $OUTPATH --cfg configs/latent.yaml --target_class $CLASS
