#!/bin/bash 
# the 2021_11_23_13_44_58_tmp is probably the best loooking
CLASS=0
SCRIPT=../basic_training/generate_save_examples.sh
PYSCRIPT=../basic_training/save_generated_images.py
MODELPATH=/home/skvara/work/counterfactual_ad/data/models/cgn_svhn2_oneclass/class_$CLASS
SAVENAME=svhn2_oneclass/class_$CLASS
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$SAVENAME
mkdir -p $OUTPATH

MODELID=20211220120700
ITERS=12000
bash $SCRIPT $MODELPATH/cgn_svhn2_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211220115208
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_svhn2_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211220115208
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_svhn2_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

