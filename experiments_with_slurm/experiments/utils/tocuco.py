import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .data import LabeledTensorDataset


def load_tocuco_dataset(tocuco_path, dataset_name, seed=0):
    tocuco_path = Path(tocuco_path)
    with open(tocuco_path / "train_masks.pkl", "rb") as train_masks_binary:
        train_masks = joblib.load(train_masks_binary)

    tocuco_datasets_path = tocuco_path / "data"

    dataset = pd.read_csv(tocuco_datasets_path / f"{dataset_name}.csv")
    dataset_seed_train_mask = train_masks[f"{dataset_name}_seed_{seed}"]
    train = dataset.loc[dataset_seed_train_mask]
    test = dataset.loc[~dataset_seed_train_mask]

    X_train = train.drop(columns=["y"])
    X_test = test.drop(columns=["y"])
    y_train = train["y"].values
    y_test = test["y"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        dataset_name,
        seed,
    )


def load_tocuco_tensor_dataset(tocuco_path, dataset_name, partition, seed=0):
    X_train, X_test, y_train, y_test, dataset_name, seed = load_tocuco_dataset(
        tocuco_path, dataset_name, seed
    )

    if partition == "train":
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        ds = LabeledTensorDataset(
            X_train_tensor,
            y_train_tensor,
            classes=np.unique(y_train),
        )
        return ds
    elif partition == "test":
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        ds = LabeledTensorDataset(
            X_test_tensor,
            y_test_tensor,
            classes=np.unique(y_test),
        )
        return ds
    else:
        raise ValueError(f"Partition {partition} not in ['train', 'test']")
