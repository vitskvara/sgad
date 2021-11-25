#!/bin/bash 
# the 2021_11_23_13_44_58_tmp is probably the best loooking
DATASET=wildlife_MNIST
bash generate_save_examples.sh mnists/experiments/cgn_${DATASET}/weights/ckp.pth $DATASET default
bash generate_save_examples.sh mnists/experiments/cgn_${DATASET}_2021_11_11_15_33_33_tmp/weights/ckp_9000.pth $DATASET 2021_11_11_15_33_33_tmp
bash generate_save_examples.sh mnists/experiments/cgn_${DATASET}_2021_11_11_15_41_02_tmp/weights/ckp_46000.pth $DATASET 2021_11_11_15_41_02_tmp
bash generate_save_examples.sh mnists/experiments/cgn_${DATASET}_2021_11_23_11_45_45_tmp/weights/ckp_46000.pth $DATASET 2021_11_23_11_45_45_tmp
bash generate_save_examples.sh mnists/experiments/cgn_${DATASET}_2021_11_23_13_44_58_tmp/weights/ckp_46000.pth $DATASET 2021_11_23_13_44_58_tmp
