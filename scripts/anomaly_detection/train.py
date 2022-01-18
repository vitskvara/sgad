import argparse
import numpy as np
import torch
import os, sys
SGADHOME='/home/skvara/work/counterfactual_ad/sgad'
sys.path.append(SGADHOME)
from sgad.utils import datadir
from sgad.cgn.train_cgn_nonmnist import load_cfg, get_cfg_defaults, merge_args_and_cfg, main
from sgad.shared.losses import PerceptualLoss

def disc_score(discriminator, x, device):
    y = torch.zeros(x.size(0),).int().to(device)
    return 1 - discriminator(x, y).data.to('cpu').numpy()

def disc_score_batched(discriminator, loader, device):
    scores = []
    labels = []
    for batch in loader:
        x = batch['ims'].to(device)
        y = batch['labels'].numpy()
        score = disc_score(discriminator, x, device)
        
        scores.append(score)
        labels.append(y)

    return np.concatenate(scores), np.concatenate(labels)

def perc_score(model, ploss, x, device):
    y_gen = torch.zeros(x.size(0),).int().to(device)
    mask, foreground, background = model(y_gen)
    x_gen = mask * foreground + (1 - mask) * background
    x_gen = x_gen.detach().to(device)
    return np.array([ploss(x_gen[i].reshape(1,*x_gen[i].size()), x[i].reshape(1,*x[i].size())).data.to('cpu').numpy() for i in range(x.size(0))])

def perc_score_batched(model, ploss, loader, device):
    scores = []
    labels = []
    for batch in loader:
        x = batch['ims'].to(device)
        y = batch['labels'].numpy()
        score = perc_score(model, ploss, x, device)
        
        scores.append(score)
        labels.append(y)

    return np.concatenate(scores), np.concatenate(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='',
                        help="path to a cfg file")
    parser.add_argument('--outpath', default='./cgn/experiments',
                        help='where the model is going to be saved')
    parser.add_argument("--save_iter", type=int, default=-1,
                        help="interval between image sampling")
    parser.add_argument("--epochs", type=int, default=-1,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=-1,
                        help="size of the batches")
    parser.add_argument("--seed", type=int, default=-1,
                        help="seed for datasplit")
    parser.add_argument("--target_class", type=int, default=-1,
                        help="target class")
    args = parser.parse_args()

    # get cfg
    cfg = load_cfg(args.cfg) if args.cfg else get_cfg_defaults()
    # add additional arguments in the argparser and in the function below
    cfg = merge_args_and_cfg(args, cfg)

    # train the model
    print(cfg)
    model, discriminator, opts, dataloaders, model_path = main(cfg)
    savepath = model_path / "scores"
    savepath.mkdir(parents=True, exist_ok=True)

    # now evaluate the anomaly scores
    tr_loader, val_loader, tst_loader = dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ploss1 = PerceptualLoss(style_wgts=cfg.LAMBDAS.PERC).to(device)
    ploss2 = PerceptualLoss(cont_wgts = [1,1,1,1], style_wgts=[1,1,1,1]).to(device)

    # save val scores
    scores_disc, labels_disc =  disc_score_batched(discriminator, val_loader, device)
    scores_perc1, labels_perc1 = perc_score_batched(model, ploss1, val_loader, device)
    scores_perc2, labels_perc2 = perc_score_batched(model, ploss2, val_loader, device)
    np.save(savepath / "val_scores_disc.npy", scores_disc)
    np.save(savepath / "val_scores_perc1.npy", scores_perc1)
    np.save(savepath / "val_scores_perc2.npy", scores_perc2)

    np.save(savepath / "val_labels_disc.npy", labels_disc)
    np.save(savepath / "val_labels_perc1.npy", labels_perc1)
    np.save(savepath / "val_labels_perc2.npy", labels_perc2)
    
    # save tst scores
    scores_disc, labels_disc =  disc_score_batched(discriminator, tst_loader, device)
    scores_perc1, labels_perc1 = perc_score_batched(model, ploss1, tst_loader, device)
    scores_perc2, labels_perc2 = perc_score_batched(model, ploss2, tst_loader, device)
    np.save(savepath / "tst_scores_disc.npy", scores_disc)
    np.save(savepath / "tst_scores_perc1.npy", scores_perc1)
    np.save(savepath / "tst_scores_perc2.npy", scores_perc2)

    np.save(savepath / "tst_labels_disc.npy", labels_disc)
    np.save(savepath / "tst_labels_perc1.npy", labels_perc1)
    np.save(savepath / "tst_labels_perc2.npy", labels_perc2)
    
    print(f"All saved to {savepath}")
