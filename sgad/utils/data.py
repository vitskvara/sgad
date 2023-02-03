import os, sys
import pickle
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
from sklearn import metrics
import urllib.request

def check_create_datadir():
    sgaddf = os.path.join(os.path.expanduser("~"), ".sgad_datapath")
    if not os.path.exists(sgaddf):
        data_path = os.path.join(os.path.expanduser("~"), "sgad_data")
        alt_path = input(f"Enter datapath for experiment storage (default {data_path}): ")
        data_path = data_path if alt_path == '' else alt_path
        with open(sgaddf, "w") as f:
            f.write(data_path)
        os.makedirs(data_path, exist_ok=True)

def datadir(*args):
    sgaddf = os.path.join(os.path.expanduser("~"), ".sgad_datapath")
    f = open(sgaddf, "r")
    data_path = f.readline()
    f.close()
    return os.path.join(data_path, *args)

def save_resize_img(x, path, n_row, sz=64):
    x = F.interpolate(x, (sz, sz))
    save_image(x.data, path, nrow=n_row, normalize=True, padding=2)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10():
    path = datadir("raw_datasets/cifar10/cifar-10-batches-py")
    fs = os.listdir(path)
    fs = filter(lambda x: x.find('_batch') != -1, fs)
    all_data = [y for y in map(lambda x: unpickle(os.path.join(path, x)), fs)]
    data = np.concatenate([x[b'data'].reshape(10000, 3, 32, 32) for x in all_data])/255
    labels = np.concatenate([x[b'labels'] for x in all_data])
    return data, labels

def _load_wmnist_part(fname, path):
    def show_progress(block_num, block_size, total_size):
        print("Downloaded "+str(round(block_num * block_size / total_size *100,2))+"%", end="\r")
    f = os.path.join(path, fname)
    if not os.path.isfile(f):
        zenodo_path = "https://zenodo.org/record/7602025/files/"
        print(f"{f} not found, should it be downloaded? (y/n)")
        response = input()
        if response in ["y", "Y"]:
            urllib.request.urlretrieve(os.path.join(zenodo_path, fname), f, show_progress)
        else:
            raise ValueError("Requested data not available.")
    return np.load(f)

def load_wildlife_mnist(train=True, denormalize=True):
    path = datadir("raw_datasets/wildlife_MNIST")
    if train:
        data = _load_wmnist_part("data.npy", path)
        labels = _load_wmnist_part("labels.npy", path)
    else:
        data = _load_wmnist_part("data_test.npy", path)
        labels = _load_wmnist_part("labels_test.npy", path)
    if denormalize:
        data = data*0.5 + 0.5
    return data, labels

def load_wildlife_mnist_split(normal_class, seed=1, train=True, denormalize=True):
    path = datadir("raw_datasets/wildlife_MNIST/training_splits")
    if train:
        tr_f = os.path.join(path, f'ac={normal_class}_seed={seed}_train_data.npy')
        if os.path.isfile(tr_f):
            tr_x = np.load(tr_f)
            tr_y = np.load(os.path.join(path, f'ac={normal_class}_seed={seed}_train_labels.npy'))
            tr_c = np.load(os.path.join(path, f'ac={normal_class}_seed={seed}_train_classes.npy'))

            val_x = np.load(os.path.join(path, f'ac={normal_class}_seed={seed}_validation_data.npy'))
            val_y = np.load(os.path.join(path, f'ac={normal_class}_seed={seed}_validation_labels.npy'))
            val_c = np.load(os.path.join(path, f'ac={normal_class}_seed={seed}_validation_classes.npy'))

            tst_x = np.load(os.path.join(path, f'ac={normal_class}_seed={seed}_test_data.npy'))
            tst_y = np.load(os.path.join(path, f'ac={normal_class}_seed={seed}_test_labels.npy'))
            tst_c = np.load(os.path.join(path, f'ac={normal_class}_seed={seed}_test_classes.npy'))
        else:
            data, labels = load_wildlife_mnist(train=True, denormalize=False)
            res = basic_data_split(data, labels, normal_class, seed=seed)
            (tr_x, tr_y, tr_c), (val_x, val_y, val_c), (tst_x, tst_y, tst_c) = res
    else:
        raise ValueError("train=False not implemented")
    if denormalize:
        tr_x = tr_x*0.5 + 0.5
        val_x = val_x*0.5 + 0.5
        tst_x = tst_x*0.5 + 0.5
    return (tr_x, tr_y, tr_c), (val_x, val_y, val_c), (tst_x, tst_y, tst_c)

def load_cifar10_split(anomaly_class, seed=1, train=True):
    path = datadir("raw_datasets/cifar10/training_splits")
    ac = anomaly_class
    if train:
        tr_x = np.load(os.path.join(path, f'ac={ac}_seed={seed}_train_data.npy'))
        tr_y = np.load(os.path.join(path, f'ac={ac}_seed={seed}_train_labels.npy'))

        val_x = np.load(os.path.join(path, f'ac={ac}_seed={seed}_validation_data.npy'))
        val_y = np.load(os.path.join(path, f'ac={ac}_seed={seed}_validation_labels.npy'))

        tst_x = np.load(os.path.join(path, f'ac={ac}_seed={seed}_test_data.npy'))
        tst_y = np.load(os.path.join(path, f'ac={ac}_seed={seed}_test_labels.npy'))
    else:
        raise ValueError("train=False not implemented")
    return (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)

def load_svhn2():
    path = datadir("raw_datasets/svhn2")
    data = np.load(os.path.join(path, "data.npy"))
    labels = np.load(os.path.join(path, "labels.npy"))
    return data, labels - 1

def train_val_test_inds(indices, ratios=(0.6,0.2,0.2), seed=None):
    if (sum(ratios) != 1.0 or len(ratios) != 3):
        raise ValueError("ratios must be a vector of length 3 that sums up to 1")

    # set seed
    rng = np.random.RandomState() if seed == None else np.random.RandomState(seed)
    _indices = rng.permutation(indices)

    # set number of samples in individual subsets
    n = len(indices)
    ns = np.cumsum(np.floor(n*np.array(ratios)).astype(int))

    # return the sets of indices
    return _indices[0:ns[0]], _indices[ns[0]+1:ns[1]], _indices[ns[1]+1:ns[2]]

def split_data_labels(data, labels, split_inds):
    tr_inds, val_inds, tst_inds = split_inds
    tr_data, val_data, tst_data = data[tr_inds], data[val_inds], data[tst_inds]
    tr_labels, val_labels, tst_labels = labels[tr_inds], labels[val_inds], labels[tst_inds]

    return (tr_data, tr_labels), (val_data, val_labels), (tst_data, tst_labels)

def basic_data_split(data, labels, normal_class, seed=None):
    """
    Normal data is split 60/20/20, anomalous 0/50/50.
    """
    normal_inds = labels == normal_class
    normal_split_inds = train_val_test_inds(np.arange(sum(normal_inds)), seed=seed)
    anomalous_split_inds = train_val_test_inds(np.arange(len(normal_inds)-sum(normal_inds)), (0,0.5,0.5), seed=seed)
    data_labels_normal = split_data_labels(data[normal_inds], labels[normal_inds], normal_split_inds)
    data_labels_anomalous = split_data_labels(data[~normal_inds], labels[~normal_inds], anomalous_split_inds)
    tr_x = np.concatenate((data_labels_normal[0][0], data_labels_anomalous[0][0]),0)
    val_x = np.concatenate((data_labels_normal[1][0], data_labels_anomalous[1][0]),0)
    tst_x = np.concatenate((data_labels_normal[2][0], data_labels_anomalous[2][0]),0)
    tr_c = np.concatenate((data_labels_normal[0][1], data_labels_anomalous[0][1]),0)
    val_c = np.concatenate((data_labels_normal[1][1], data_labels_anomalous[1][1]),0)
    tst_c = np.concatenate((data_labels_normal[2][1], data_labels_anomalous[2][1]),0)
    tr_y = np.zeros(len(tr_c))
    val_y = np.concatenate((np.zeros(len(data_labels_normal[1][1])), np.ones(len(data_labels_anomalous[1][1]))),0)
    tst_y = np.concatenate((np.zeros(len(data_labels_normal[2][1])), np.ones(len(data_labels_anomalous[2][1]))),0)
    return (tr_x, tr_y, tr_c), (val_x, val_y, val_c), (tst_x, tst_y, tst_c)

def compute_auc(labels, scores, pos_label=1):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=pos_label)
    return metrics.auc(fpr, tpr)
