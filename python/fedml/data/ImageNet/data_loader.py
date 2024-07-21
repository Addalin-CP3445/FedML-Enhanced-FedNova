import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


from .datasets import ImageNet
from .datasets import ImageNet_truncated
from .datasets_hdf5 import ImageNet_hdf5
from .datasets_hdf5 import ImageNet_truncated_hdf5


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

def partition_data_dirichlet(labels, client_number, alpha):
    """
    Partition data indices using Dirichlet distribution for non-iid partitioning.

    :param labels: Array of labels in the dataset.
    :param client_number: Number of clients to partition data for.
    :param alpha: Dirichlet distribution parameter.
    :return: A dictionary where keys are client indices and values are lists of data indices.
    """
    np.random.seed(42)  # for reproducibility

    label_distribution = np.bincount(labels)
    num_classes = len(label_distribution)
    logging.info("LABELs: " + str(labels))
    logging.info("num_classes: " + str(num_classes))

    # Compute the label proportions for each client
    label_proportions = np.random.dirichlet([alpha] * client_number, num_classes)
    client_dataidx_map = {i: [] for i in range(client_number)}

    print("Label distribution:", label_distribution)
    print("Label proportions:", label_proportions)

    # Assign indices to each client based on the label proportions
    for label in range(num_classes):
        indices = []
        for i, l in enumerate(labels):
            if l == label:
                indices.append(i)

        #indices = np.where(labels == label)[0]
        np.random.shuffle(indices)
        print(f"Label {label}, labels: {labels}")
        print(f"Indices: {indices}")
        proportions = label_proportions[label]
        print(f"Label {label}, indices count: {len(indices)}, proportions: {proportions}")
        start_idx = 0
        for client_id, proportion in enumerate(proportions):
            num_samples = int(proportion * len(indices))
            client_dataidx_map[client_id].extend(indices[start_idx:start_idx + num_samples])
            start_idx += num_samples

    for client_id, indices in client_dataidx_map.items():
        print(f"Client {client_id}: {len(indices)} samples")

    return client_dataidx_map

def _data_transforms_ImageNet():
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    image_size = 224
    train_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, valid_transform


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(
    dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test
):
    return get_dataloader_test_ImageNet(
        datadir, train_bs, test_bs, dataidxs_train, dataidxs_test
    )


def get_dataloader_ImageNet_truncated(
    imagenet_dataset_train,
    imagenet_dataset_test,
    train_bs,
    test_bs,
    dataidxs=None,
    net_dataidx_map=None,
):
    """
    imagenet_dataset_train, imagenet_dataset_test should be ImageNet or ImageNet_hdf5
    """
    if type(imagenet_dataset_train) == ImageNet:
        dl_obj = ImageNet_truncated
    elif type(imagenet_dataset_train) == ImageNet_hdf5:
        dl_obj = ImageNet_truncated_hdf5
    else:
        raise NotImplementedError()

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(
        imagenet_dataset_train,
        dataidxs,
        net_dataidx_map,
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = dl_obj(
        imagenet_dataset_test,
        dataidxs=None,
        net_dataidx_map=None,
        train=False,
        transform=transform_test,
        download=False,
    )

    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


def get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = ImageNet

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs,
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = dl_obj(
        datadir, dataidxs=None, train=False, transform=transform_test, download=False
    )

    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


def get_dataloader_test_ImageNet(
    datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None
):
    dl_obj = ImageNet

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs_train,
        train=True,
        transform=transform_train,
        download=True,
    )
    test_ds = dl_obj(
        datadir,
        dataidxs=dataidxs_test,
        train=False,
        transform=transform_test,
        download=True,
    )

    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


def distributed_centralized_ImageNet_loader(
    dataset, data_dir, world_size, rank, batch_size
):
    """
    Used for generating distributed dataloader for
    accelerating centralized training
    """

    train_bs = batch_size
    test_bs = batch_size

    transform_train, transform_test = _data_transforms_ImageNet()
    if dataset == "ILSVRC2012":
        train_dataset = ImageNet(
            data_dir=data_dir, dataidxs=None, train=True, transform=transform_train
        )

        test_dataset = ImageNet(
            data_dir=data_dir, dataidxs=None, train=False, transform=transform_test
        )
    elif dataset == "ILSVRC2012_hdf5":
        train_dataset = ImageNet_hdf5(
            data_dir=data_dir, dataidxs=None, train=True, transform=transform_train
        )

        test_dataset = ImageNet_hdf5(
            data_dir=data_dir, dataidxs=None, train=False, transform=transform_test
        )

    train_sam = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sam = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_dl = data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        sampler=train_sam,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        sampler=test_sam,
        pin_memory=True,
        num_workers=4,
    )

    class_num = 1000

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    return train_data_num, test_data_num, train_dl, test_dl, None, None, None, class_num


def load_partition_data_ImageNet(
    dataset,
    data_dir,
    partition_method=None,
    partition_alpha=None,
    client_number=100,
    batch_size=10,
):

    if dataset == "ILSVRC2012":
        temp_dataset = ImageNet(data_dir=data_dir, dataidxs=None, train=True)
        test_dataset = ImageNet(data_dir=data_dir, dataidxs=None, train=False)
        #for idx, item in enumerate(temp_dataset.all_data):
        #    logging.info(item)
        labels = [item[1] for item in temp_dataset.all_data]
    elif dataset == "ILSVRC2012_hdf5":
        temp_dataset = ImageNet_hdf5(data_dir=data_dir, dataidxs=None, train=True)
        test_dataset = ImageNet_hdf5(data_dir=data_dir, dataidxs=None, train=False)
        labels = temp_dataset.all_data_hdf5.dlabel
        
    # Debug: Print the labels array
    print("Labels array:", labels)
    print("Number of labels:", len(labels))

    if partition_method == 'hetro' and partition_alpha is not None:
        dataidxs = partition_data_dirichlet(labels, client_number, partition_alpha)
        print("Data indices (dataidxs):")
        for k, v in dataidxs.items():
            print(f"Client {k}: {len(v)} samples")
    else:
        raise ValueError("Unsupported partition method or missing partition alpha")

    if dataset == "ILSVRC2012":
        train_dataset = ImageNet(data_dir=data_dir, dataidxs=dataidxs, train=True)
    elif dataset == "ILSVRC2012_hdf5":
        train_dataset = ImageNet_hdf5(data_dir=data_dir, dataidxs=dataidxs, train=True)

    net_dataidx_map = train_dataset.get_net_dataidx_map()

    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    class_num_dict = train_dataset.get_data_local_num_dict()

    # train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)

    train_data_global, test_data_global = get_dataloader_ImageNet_truncated(
        train_dataset,
        test_dataset,
        train_bs=batch_size,
        test_bs=batch_size,
        dataidxs=None,
        net_dataidx_map=None,
    )

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        if dataidxs is not None:
            if client_idx not in dataidxs:
                raise KeyError(f"Client index {client_idx} is not in dataidxs")
            local_data_num = len(dataidxs[client_idx])
            data_local_num_dict[client_idx] = local_data_num

        train_data_local, test_data_local = get_dataloader_ImageNet_truncated(
            train_dataset,
            test_dataset,
            train_bs=batch_size,
            test_bs=batch_size,
            dataidxs=dataidxs[client_idx],
            net_dataidx_map=net_dataidx_map,
        )

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    logging.info("data_local_num_dict: %s" % data_local_num_dict)
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


if __name__ == "__main__":
    # data_dir = '/home/datasets/imagenet/ILSVRC2012_dataset'
    data_dir = "/home/datasets/imagenet/imagenet_hdf5/imagenet-shuffled.hdf5"

    client_number = 100
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_ImageNet(
        None,
        data_dir,
        partition_method=None,
        partition_alpha=None,
        client_number=client_number,
        batch_size=10,
    )

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

    i = 0
    for data, label in train_data_global:
        print(data)
        print(label)
        i += 1
        if i > 5:
            break
    print("=============================\n")

    for client_idx in range(client_number):
        i = 0
        for data, label in train_data_local_dict[client_idx]:
            print(data)
            print(label)
            i += 1
            if i > 5:
                break
