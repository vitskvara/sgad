#!/bin/bash 
# the 2021_11_23_13_44_58_tmp is probably the best loooking
CLASS=1
SCRIPT=../basic_training/generate_save_examples.sh
PYSCRIPT=../basic_training/save_generated_images.py
MODELPATH=/home/skvara/work/counterfactual_ad/data/models/cgn_cifar_oneclass/class_$CLASS
SAVENAME=cifar10_oneclass/class_$CLASS
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$SAVENAME
mkdir -p $OUTPATH

MODELID=20211208173040
ITERS=12000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211208173708
ITERS=12000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth
