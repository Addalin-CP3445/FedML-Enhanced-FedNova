import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from collections import OrderedDict
from PIL import Image

from .without_reload import CIFAR10_truncated, CIFAR10_truncated_WO_reload

# generate the non-IID distribution for all methods
def read_data_distribution(filename="./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt",):
    distribution = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0]:
                tmp = x.split(":")
                if "{" == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(",", ""))
    return distribution


def read_net_dataidx_map(filename="./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt",):
    net_dataidx_map = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0] and "]" != x[0]:
                tmp = x.split(":")
                if "[" == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(",")
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class LaplaceNoiseConfig:
    def __init__(self, enable=False, epsilon=0.1, sensitivity=1.0):
        self.enable = enable
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.mechanism = 'laplace'


class AddLaplaceNoise(object):
    def __init__(self, epsilon, sensitivity):
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def __call__(self, img):
        # img = np.array(img)

        # #if img.dtype != np.uint8:
        # #    img = img.astype(np.uint8)

        # scale = self.sensitivity / self.epsilon

        # # Generate Laplace noise
        # noise = np.random.laplace(0, scale, img.shape)

        # # Add noise to the image
        # noisy_image = img + noise

        # # Clip values to be in the valid range [0, 255]
        # noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        # noisy_image = np.transpose(noisy_image, (1, 2, 0))

        # # Debugging prints
        # # print(f"Noisy image shape: {noisy_image.shape}, dtype: {noisy_image.dtype}")

        # # Ensure the shape and dtype are correct
        # if len(noisy_image.shape) != 3 or noisy_image.shape[2] not in [1, 3]:
        #     raise ValueError(f"Unexpected noisy image shape: {noisy_image.shape}")
        # if noisy_image.dtype != np.uint8:
        #     raise TypeError(f"Unexpected noisy image dtype: {noisy_image.dtype}")

        # Image.fromarray(noisy_image)  # Convert back to PIL Image
        # return transforms.ToTensor()(noisy_image)
        
        img = np.array(img).astype(np.float32) / 255.0  # Normalize image to [0, 1] range
        noise = np.random.laplace(0, self.sensitivity / self.epsilon, img.shape)
        noisy_image = img + noise
        noisy_image = np.clip(noisy_image, 0, 1)  # Clip to [0, 1] range
        noisy_image = (noisy_image * 255).astype(np.uint8)  # Convert back to [0, 255]
        return Image.fromarray(noisy_image)
    
    def __repr__(self):
        return self.__class__.__name__ + '(epsilon={0}, sensitivity={1})'.format(self.epsilon, self.sensitivity)

class GaussianNoiseConfig:
    def __init__(self, enable=False, epsilon=0.1, sensitivity=1.0, delta=0.00001):
        self.enable = enable
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.delta = delta
        self.mechanism = 'gaussian'
        
class AddGaussianNoise(object):
    def __init__(self, epsilon, sensitivity, delta):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.delta = delta

    def __call__(self, img):
        scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        img = np.array(img).astype(np.float32) / 255.0  # Normalize image to [0, 1] range
        noise = np.random.normal(0, scale, img.shape)
        noisy_image = img + noise
        noisy_image = np.clip(noisy_image, 0, 1)  # Clip to [0, 1] range
        noisy_image = (noisy_image * 255).astype(np.uint8)  # Convert back to [0, 255]
        return Image.fromarray(noisy_image)
    
    def __repr__(self):
        return self.__class__.__name__ + '(epsilon={0}, sensitivity={1}, delta={2})'.format(self.epsilon, self.sensitivity, self.delta)

def _data_transforms_cifar10(noise_config=None):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if noise_config and noise_config.enable and noise_config.mechanism == 'laplace':
        train_transform.append(AddLaplaceNoise(noise_config.epsilon, noise_config.sensitivity))
    elif noise_config and noise_config.enable and noise_config.mechanism == 'gaussian':
        train_transform.append(AddGaussianNoise(noise_config.epsilon, noise_config.sensitivity, noise_config.delta))

    train_transform.append(transforms.ToTensor())
    train_transform.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
    train_transform.append(Cutout(16))

    train_transform = transforms.Compose(train_transform)

    valid_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    return train_transform, valid_transform


def load_cifar10_data(datadir, process_id, synthetic_data_url, private_local_data,noise_config = None, resize=32, augmentation=True, data_efficient_load=False):
    train_transform, test_transform = _data_transforms_cifar10(noise_config)

    is_download = True;

    if data_efficient_load:
        cifar10_train_ds = CIFAR10(datadir, train=True, download=True, transform=train_transform)
        cifar10_test_ds = CIFAR10(datadir, train=False, download=True, transform=test_transform)
    else:
        cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=is_download, transform=train_transform)
        cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=is_download, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets

    return (X_train, y_train, X_test, y_test, cifar10_train_ds, cifar10_test_ds)


def partition_data(dataset, datadir, partition, n_nets, alpha, process_id, synthetic_data_url, private_local_data, noise_config = None):
    np.random.seed(10)
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test, cifar10_train_ds, cifar10_test_ds = load_cifar10_data(datadir, process_id, synthetic_data_url, private_local_data, noise_config)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = "./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt"
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = "./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt"
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, cifar10_train_ds, cifar10_test_ds


# for centralized training
def get_dataloader(
    dataset,
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    noise_config = None,
    data_efficient_load=False,
    full_train_dataset=None,
    full_test_dataset=None,
):
    return get_dataloader_CIFAR10(
        datadir,
        train_bs,
        test_bs,
        dataidxs,
        noise_config,
        data_efficient_load=data_efficient_load,
        full_train_dataset=full_train_dataset,
        full_test_dataset=full_test_dataset,
    )


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR10(
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    noise_config=None,
    data_efficient_load=False,
    full_train_dataset=None,
    full_test_dataset=None,
):
    transform_train, transform_test = _data_transforms_cifar10(noise_config)

    if data_efficient_load:
        dl_obj = CIFAR10_truncated_WO_reload
        train_ds = dl_obj(
            datadir, dataidxs=dataidxs, train=True, transform=transform_train, full_dataset=full_train_dataset
        )
        test_ds = dl_obj(datadir, train=False, transform=transform_test, full_dataset=full_test_dataset)
    else:
        dl_obj = CIFAR10_truncated
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True,)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True,)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(
    process_id,
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    data_efficient_load=True,
):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
        cifar10_train_ds,
        cifar10_test_ds,
    ) = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            data_efficient_load=True,
            full_train_dataset=cifar10_train_ds,
            full_test_dataset=cifar10_test_ds,
        )
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            data_efficient_load=True,
            full_train_dataset=cifar10_train_ds,
            full_test_dataset=cifar10_test_ds,
        )
        logging.info(
            "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (process_id, len(train_data_local), len(test_data_local))
        )
        train_data_global = None
        test_data_global = None
    return (
        train_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        class_num,
    )


def efficient_load_partition_data_cifar10(
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    process_id=0,
    synthetic_data_url="",
    private_local_data="",
    n_proc_in_silo=0,
    data_efficient_load=True,
    noise_config=None, 
):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
        cifar10_train_ds,
        cifar10_test_ds,
    ) = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha, process_id, synthetic_data_url, private_local_data, noise_config)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        dataset,
        data_dir,
        batch_size,
        batch_size,
        dataidxs=None,
        noise_config=noise_config,
        data_efficient_load=True,
        full_train_dataset=cifar10_train_ds,
        full_test_dataset=cifar10_test_ds,
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            noise_config=noise_config,
            data_efficient_load=True,
            full_train_dataset=cifar10_train_ds,
            full_test_dataset=cifar10_test_ds,
        )
        logging.info(
            "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (client_idx, len(train_data_local), len(test_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
