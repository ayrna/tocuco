import os
import joblib
import pandas as pd


def stream_tocuco_datasets(tmp_tocuco_path, seeds=5):
    with open(os.path.join(tmp_tocuco_path, "train_masks.pkl"), "rb") as train_masks_binary:
        train_masks = joblib.load(train_masks_binary)

    tocuco_datasets_path = os.path.join(tmp_tocuco_path, "data")

    for dataset_name in os.listdir(tocuco_datasets_path):
        dataset = pd.read_csv(os.path.join(tocuco_datasets_path, dataset_name))
        for seed in range(seeds):
            dataset_name_without_extension = dataset_name.split(".")[0]

            dataset_seed_train_mask = train_masks[f"{dataset_name_without_extension}_seed_{seed}"]
            train = dataset.loc[dataset_seed_train_mask]
            test = dataset.loc[~dataset_seed_train_mask]

            X_train = train.drop(columns=["y"]).to_numpy()
            X_test = test.drop(columns=["y"]).to_numpy()
            y_train = train["y"].to_numpy()
            y_test = test["y"].to_numpy()

            yield (X_train, X_test, y_train, y_test, dataset_name_without_extension, seed)
