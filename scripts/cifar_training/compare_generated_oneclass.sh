#!/bin/bash 
# the 2021_11_23_13_44_58_tmp is probably the best loooking
CLASS=2
SCRIPT=../basic_training/generate_save_examples.sh
PYSCRIPT=../basic_training/save_generated_images.py
MODELPATH=/home/skvara/work/counterfactual_ad/data/models/cgn_cifar_oneclass/class_$CLASS
SAVENAME=cifar10_oneclass/class_$CLASS
OUTPATH=/home/skvara/work/counterfactual_ad/data/cgn_generated/$SAVENAME
mkdir -p $OUTPATH

MODELID=20211209075334
ITERS=12000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209080619
ITERS=12000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209081907
ITERS=12000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209083202
ITERS=12000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209084448
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209085124
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209085801
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209090439
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209091117
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209091755
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209092431
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth

MODELID=20211209093103
ITERS=4000
bash $SCRIPT $MODELPATH/cgn_cifar10_model_id-$MODELID/weights/ckp_$ITERS.pth $SAVENAME
python $PYSCRIPT $OUTPATH/${MODELID}_${ITERS}_counterfactual.pth
