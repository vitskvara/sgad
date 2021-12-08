#!/bin/bash 
# the 2021_11_23_13_44_58_tmp is probably the best loooking
DATASET=cifar10
SCRIPT=../basic_training/generate_save_examples.sh
PYSCRIPT=../basic_training/save_generated_images.py
MODELPATH=/home/skvara/work/counterfactual_ad/data/models/cgn_cifar
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$DATASET

MODELID=20211207205846 # 64 batchsize, 64 latentsize
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207212808 # 64 batchsize, 64 latentsize
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207215734 # 64 batchsize, 64 latentsize
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207152404 # 60 epochs - this does not help
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_132000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

# lots of cars
MODELID=20211207162506 # 60 epochs
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_132000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207172542 # 60 epochs
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_132000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207182659 # 64 batchsize
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207185623 # 64 batchsize
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207192542 # 64 batchsize
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207195508 # linear disc, 64batchsize - this sucks
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207203739 # linear disc, 64batchsize
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207201625 # linear disc, 64batchsize
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

# this one looks quite good
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-20211207120826/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/20211207120826_counterfactual.pth

bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-20211207130933/weights/ckp_32000.pth $DATASET
python $PYSCRIPT $OUTPATH/20211207130933_counterfactual.pth

bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-20211207133522/weights/ckp_64000.pth $DATASET
python $PYSCRIPT $OUTPATH/20211207133522_counterfactual.pth

MODELID=20211207140553
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_64000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

MODELID=20211207143634
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_64000.pth $DATASET
python $PYSCRIPT $OUTPATH/${MODELID}_counterfactual.pth

