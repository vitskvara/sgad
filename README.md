# Shape Guided Anomaly Detection
This project aims to detect anomalies in images on three different levels - shape, foreground and background texture.

# Instalation
Preferably into a specific python environment.
```
git clone https://github.com/vitskvara/sgad.git
cd sgad
pip install .
```
*TODO* You may also need some basic data, which are not yet available here.

# Test run
This tests basic one class runs.
```
cd sgad/test
./train_cgn_cifar10.sh
./train_cgn_other.sh
```
If you get `[SSL: CERTIFICATE_VERIFY_FAILED]` when the package tries to download the vgg16 weights, you can try to download them manually to the indicated directory, e.g. `wget https://download.pytorch.org/models/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints`.