import json
from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset
from torchvision.datasets import ImageFolder


def get_dataset_info(data_dir, dataset_name):
    data_dir = Path(data_dir)
    with open(data_dir / "datasets.json", "r") as f:
        datasets_info = json.load(f)

    if dataset_name not in datasets_info:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in datasets.json")

    dataset_info = datasets_info[dataset_name]

    if "train" in dataset_info:
        dataset_info["train"] = data_dir / dataset_info["train"]

    if "val" in dataset_info:
        dataset_info["val"] = data_dir / dataset_info["val"]

    if "test" in dataset_info:
        dataset_info["test"] = data_dir / dataset_info["test"]

    return dataset_info


def load_data(*, data_dir: str, dataset: str, partition: str, seed: int) -> Dataset:
    if partition not in ["train", "test"]:
        raise ValueError(f"Partition {partition} not in ['train', 'test']")

    # TOCUCO datasets
    if dataset.startswith("tocuco_"):
        from experiments.utils.tocuco import load_tocuco_tensor_dataset

        return load_tocuco_tensor_dataset(
            str(Path(data_dir) / "TOCUCO"),
            dataset.replace("tocuco_", ""),
            partition,
            seed,
        )

    try:
        dataset_info = get_dataset_info(data_dir, dataset)
        data_path = dataset_info[partition]
        data_type = dataset_info["type"]

        if data_type == "image":
            from torch import float32 as torch_float32
            from torchvision import disable_beta_transforms_warning
            from torchvision.datasets import ImageFolder
            from torchvision.transforms import v2 as transforms

            disable_beta_transforms_warning()

            transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch_float32, scale=True),
                ]
            )

            dataset = ImageFolder(data_path, transform=transform)

            print(f"Loaded {partition} data from {data_path}.")
            return dataset
        else:
            raise NotImplementedError(f"Data type {data_type} not implemented")

    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {dataset} not found in datasets.json")


def create_dataset_resample(
    train_dataset: Dataset, test_dataset: Dataset, random_state=0
) -> Tuple[Dataset, Dataset]:
    if random_state == 0:
        return train_dataset, test_dataset

    dataset = MyConcatDataset([train_dataset, test_dataset])
    test_size = len(test_dataset)

    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    sss_splits = list(sss.split(X=np.zeros(len(dataset)), y=dataset.targets))
    train_idx, test_idx = sss_splits[0]

    new_train_dataset = MySubset(dataset, train_idx)
    new_test_dataset = MySubset(dataset, test_idx)

    return new_train_dataset, new_test_dataset


class MySubset(Subset):
    """
    Subset of a dataset at specified indices. Includes targets if they are
    available in the original dataset. Soporta datasets tabulares e imágenes de forma segura.
    """

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

        if hasattr(dataset, "targets"):
            self.targets = [dataset.targets[i] for i in indices]
        else:
            self.targets = []


    @property
    def classes(self):
        return np.unique(self.targets)

    @property
    def data(self):
        if hasattr(self.dataset, "data"):
            return self.dataset.data[self.indices]
        raise AttributeError("The dataset does not support direct access to data (likely an ImageFolder).")

    @property
    def shape(self):
        if hasattr(self.dataset, "data"):
            return self.data.shape
        raise AttributeError("It is not possible to obtain the shape of an image dataset without loading it.")

    @property
    def ndim(self):
        if hasattr(self.dataset, "data"):
            return self.data.ndim
        raise AttributeError("It is not possible to obtain the shape of an image dataset without loading it.")


class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)

    @property
    def targets(self):
        return [target for dataset in self.datasets for target in dataset.targets]

    @property
    def classes(self):
        return np.unique(self.targets)


def get_dataset_fold(dataset, *, fold, n_folds, random_state=0):
    if n_folds is None or n_folds <= 1 or fold is None:
        raise ValueError("n_folds must be greater than 1 and fold must be specified")

    from sklearn.model_selection import StratifiedKFold

    data_length = len(dataset)
    targets = dataset.targets

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    skf_splits = list(skf.split(X=np.zeros(data_length), y=targets))
    train_idx, val_idx = skf_splits[fold]

    train_data = MySubset(dataset, train_idx)
    val_data = MySubset(dataset, val_idx)

    return train_data, val_data


def get_dataset_holdout(dataset, *, test_size, random_state=0):
    from sklearn.model_selection import StratifiedShuffleSplit

    data_length = len(dataset)
    targets = dataset.targets

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    sss_splits = list(sss.split(X=np.zeros(data_length), y=targets))
    train_idx, test_idx = sss_splits[0]

    train_data = MySubset(dataset, train_idx)
    test_data = MySubset(dataset, test_idx)

    return train_data, test_data


class LabeledTensorDataset(TensorDataset):
    def __init__(self, data_tensor, target_tensor, classes):
        super().__init__(data_tensor, target_tensor)
        self.data = data_tensor
        self.targets = target_tensor.tolist()
        self.classes = classes

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y.item()