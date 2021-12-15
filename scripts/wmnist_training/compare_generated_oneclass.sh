#!/bin/bash 
# the 2021_11_23_13_44_58_tmp is probably the best loooking
CLASS=0
SCRIPT=../basic_training/generate_save_examples.sh
PYSCRIPT=../basic_training/save_generated_images.py
MODELPATH=/home/skvara/work/counterfactual_ad/data/models/cgn_wmnist_oneclass/class_$CLASS
SAVENAME=wmnist_oneclass/class_$CLASS
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$SAVENAME
mkdir -p $OUTPATH

MODELID=20211214142450
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214144941
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214151453
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214154004
ITERS=24000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214160708
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214162046
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214163454
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214164759
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214170037
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214171401
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214172723
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211214174117
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_wildlife_mnist_model_id-$MODELID/weights/cgn_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth
