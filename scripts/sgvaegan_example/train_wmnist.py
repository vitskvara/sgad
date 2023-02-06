import sgad
import numpy as np
import os
import torch
from torch import nn
from pathlib import Path
import argparse
from sklearn.metrics import roc_auc_score

from sgad.sgvae import SGVAEGAN
from sgad.utils import load_wildlife_mnist_split, datadir
from sgad.utils import save_cfg, load_cfg

# arg parser
parser = argparse.ArgumentParser(
                    prog = 'Train a basic SGVAEGAN model on Wildlife MNIST data.')
parser.add_argument('--normal_class', type=int, default=2, help="label of the normal class (0-9)")
parser.add_argument('--split_seed', type=int, default=4, help="seed with which the data is split")
parser.add_argument('--zdim', type=int, default=32,	help="latent space size")
parser.add_argument('--n_epochs', type=int, default=50, help="no. epochs")
args = parser.parse_args()

# setup
normal_class = args.normal_class
seed = args.split_seed
n_epochs = args.n_epochs

# setup path where the model outputs will be saved
outpath = Path(datadir("test_models/"))
outpath.mkdir(parents=True, exist_ok=True)
nfiles = len(os.listdir(outpath))
outpath = os.path.join(outpath, f"run_{nfiles+1}")

# load the data - note that the images are in [-1,1] range
data = load_wildlife_mnist_split(normal_class, seed=seed, train=True, denormalize=False)
(tr_x, tr_y, tr_c), (val_x, val_y, val_c), (tst_x, tst_y, tst_c) = data

# use the defaults for training on wildlife mnist
model = SGVAEGAN(zdim=args.zdim)
losses, best_model, best_epoch = model.fit(tr_x, n_epochs=n_epochs, save_path=outpath, save_weights=False)

# now compute basic anomaly detection metrics
print("\nTraining of the base model finished...\n")
discriminator_scores = model.discriminator_score(tst_x)
reconstruction_scores = model.reconstruction_score(tst_x)
feature_matching_scores = model.feature_matching_score(tst_x)
discriminator_auc = roc_auc_score(tst_y, discriminator_scores)
reconstruction_auc = roc_auc_score(tst_y, reconstruction_scores)
feature_matching_auc = roc_auc_score(tst_y, feature_matching_scores)
print(f"Discriminator-based AUC={discriminator_auc}")
print(f"Reconstruction-based AUC={reconstruction_auc}")
print(f"Feature-matching-based AUC={feature_matching_auc}")

############# ALPHA SCORES ###############
k = 31
print("Fitting the alpha params...")
model.fit_alpha(tr_x, val_x, val_y, k, beta0=10.0, 
            alpha0=np.array([1.0, 0, 0, 0, 0]), # sometimes this helps with convergence
            init_alpha = np.array([1.0, 1.0, 0, 0, 0]) 
 	)
tst_scores_alpha = model.predict_with_alpha(model, tst_x)
alpha_auc = roc_auc_score(tst_y, tst_scores_alpha)
print(f"Alpha-based AUC={alpha_auc}")









from sklearn.neighbors import NearestNeighbors
from sgad.sgvae import RobustLogisticRegression
k = 31

# alpha scores - first compute the encodings
tr_encodings = model.encode_mean_batched(tr_x)
val_encodings = model.encode_mean_batched(val_x)

# now compute the knn scores - for each latent separately
def knn_score(knn_model, x):
	return knn_model.kneighbors(x)[0].mean(1)

#
knn_models = [NearestNeighbors(n_neighbors=k) for _ in range(3)]
[m.fit(encodings, tr_y) for (m,encodings) in zip(knn_models, tr_encodings)]

# now compute the alpha scores on the validation data
rec_scores = model.reconstruction_score(val_x)
disc_scores = model.discriminator_score(val_x)
knn_scores = [knn_score(m, encodings) for (m,encodings) in zip(knn_models, val_encodings)]
val_scores = np.vstack((rec_scores, disc_scores, *knn_scores)).transpose()

beta = 10/sum(val_y)
init_alpha = np.array([1.0, 1.0, 0, 0, 0]) # sometimes this helps with better convergence
alpha0 = np.array([1.0, 0, 0, 0, 0])
lr_model = RobustLogisticRegression(val_scores.shape[1], alpha=init_alpha, beta=beta, alpha0=alpha0)
lr_model.fit(val_scores, val_y, scale=True, early_stopping=True)
alpha = lr_model.alpha.detach().numpy()


# predict
tst_encodings = model.encode_mean_batched(tst_x)
rec_scores = model.reconstruction_score(tst_x)
disc_scores = model.discriminator_score(tst_x)
knn_scores = [knn_score(m, encodings) for (m,encodings) in zip(knn_models, tst_encodings)]
tst_scores = np.vstack((rec_scores, disc_scores, *knn_scores)).transpose()
tst_scores = lr_model.scaler_transform(tst_scores)
tst_scores_alpha = np.matmul(tst_scores, alpha)

roc_auc_score(tst_y, tst_scores_alpha)
