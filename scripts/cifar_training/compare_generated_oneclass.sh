#!/bin/bash 
# the 2021_11_23_13_44_58_tmp is probably the best loooking
CLASS=1
SCRIPT=../basic_training/generate_save_examples.sh
PYSCRIPT=../basic_training/save_generated_images.py
MODELPATH=/home/skvara/work/counterfactual_ad/data/models/cgn_cifar_oneclass/class_$CLASS
SAVENAME=cifar10_oneclass/class_$CLASS
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$SAVENAME
mkdir -p $OUTPATH

MODELID=20211214141851
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214144054
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214150616
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214153141
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214155717
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214160931
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214162315
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214163740
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214165055
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214170437
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214171824
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214173222
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth
