import argparse
import warnings
from tqdm import trange
import os
import torch
import repackage
repackage.up()

from cgn.train_cgn import CGN
from cgn.dataloader import get_dataloaders
from utils import load_cfg

def generate_cf_dataset(model, path, dataset_size, no_cfs, device):
    x, y = [], []
    model.batch_size = 100
    n_classes = 10

    total_iters = int(dataset_size // model.batch_size // no_cfs)
    for _ in trange(total_iters):

        # generate initial mask
        y_gen = torch.randint(n_classes, (model.batch_size,)).to(device)
        mask, _, _ = model(y_gen)

        # generate counterfactuals, i.e., same masks, foreground/background vary
        for _ in range(no_cfs):
            _, foreground, background = model(y_gen, counterfactual=True)
            x_gen = mask * foreground + (1 - mask) * background

            x.append(x_gen.detach().cpu())
            y.append(y_gen.detach().cpu())

    dataset = [torch.cat(x), torch.cat(y)]
    print(f"x shape {dataset[0].shape}, y shape {dataset[1].shape}")
    torch.save(dataset, path)
    print(f"saved generated data to {path}")

def generate_dataset(dl, path):
    x, y = [], []
    for data in dl:
        x.append(data['ims'].cpu())
        y.append(data['labels'].cpu())

    dataset = [torch.cat(x), torch.cat(y)]

    print(f"Saving to {path}")
    print(f"x shape: {dataset[0].shape}, y shape: {dataset[1].shape}")
    torch.save(dataset, 'mnists/data/' + path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['colored_MNIST', 'double_colored_MNIST', 'wildlife_MNIST'],
                        help='Name of the dataset. Make sure the name and the weight_path match')
    parser.add_argument('--weight_path', default='',
                        help='Provide path to .pth of the model')
    parser.add_argument('--dataset_size', type=float, default=5e4,
                        help='Size of the dataset. For counterfactual data: the more the better.')
    parser.add_argument('--no_cfs', type=int, default=1,
                        help='How many counterfactuals to sample per datapoint')
    parser.add_argument('--outpath', default='.',
                        help='Where to save the generated data.')
    args = parser.parse_args()
    print(args)

    assert args.weight_path or args.dataset, "Supply dataset name or weight path."
    if args.weight_path: assert args.dataset, "Also supply the dataset type."
    os.makedirs(args.outpath, exist_ok=True)

    # Generate the dataset
    if not args.weight_path:
        # get dataloader
        dl_train, dl_test = get_dataloaders(args.dataset, batch_size=1000, workers=8)

        # generate
        generate_dataset(dl=dl_train, path=os.path.join(args.outpath, args.dataset + '_train.pth'))
        generate_dataset(dl=dl_test, path=os.path.join(args.outpath, args.dataset + '_test.pth'))

    # Generate counterfactual dataset
    else:
        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CGN()
        model.load_state_dict(torch.load(args.weight_path, 'cpu'))
        model.to(device).eval()

        # generate
        print(f"Generating the counterfactual {args.dataset} of size {args.dataset_size}")
        generate_cf_dataset(cgn=model, path=os.path.join(args.outpath, args.dataset + '_counterfactual.pth'),
                            dataset_size=args.dataset_size, no_cfs=args.no_cfs,
                            device=device)
