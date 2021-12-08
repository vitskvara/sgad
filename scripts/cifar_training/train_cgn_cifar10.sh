#!/bin/bash 
DATADIR=/home/skvara/work/counterfactual_ad/data
SGADHOME=/home/skvara/work/counterfactual_ad/sgad/sgad
#python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/cgn_cfg.yaml
#python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/cgn_cfg.yaml
#python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/cgn_cfg.yaml

python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/cgn_cfg.yaml
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/cgn_cfg.yaml
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/cgn_cfg.yaml

python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/batchsize.yaml
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/batchsize.yaml
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/batchsize.yaml

python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/disc.yaml
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/disc.yaml
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/disc.yaml

python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/latent.yaml
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/latent.yaml
python $SGADHOME/cgn/train_cgn_nonmnist.py --seed 1 --outpath $DATADIR/models/cgn_cifar --cfg configs/latent.yaml