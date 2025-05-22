import torch
import torchvision


def get_dataset(dataset, data_dir, transform, train=True, download=True, debug_subset_size=None, train_feature=True, dir=""):
    if dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='test' if train == True else 'val', transform=transform,
                                                download=download)
    # elif dataset == 'AID':
    #     if train:
    #         dataset = torchvision.datasets.ImageFolder('dataset/AID_linear/train_0.2', transform=transform)
    #     else:
    #         dataset = torchvision.datasets.ImageFolder('dataset/AID_linear/test', transform=transform)

    elif dataset in ['EuroSAT', 'UCMerced', 'AID', 'RESISC45']:
        if train_feature:
            if train:
                dataset = torchvision.datasets.ImageFolder(dir, transform=transform)
            else:
                # dataset = RSDataset_nolabel('dataset/AID_list/valid.txt', transform=transform)
                dataset = torchvision.datasets.ImageFolder(dir, transform=transform)
        else:
            if train:
                dataset = torchvision.datasets.ImageFolder(dir, transform=transform)
            else:
                # dataset = RSDataset_nolabel('dataset/AID_list/valid.txt', transform=transform)
                dataset = torchvision.datasets.ImageFolder(dir, transform=transform)

    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size))  # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset
