import os
import os.path as osp

from datasets import Dataset, Audio
import pandas as pd
import os.path as osp
from datasets import Audio


def load_train_dataset():
    df = pd.read_csv("data/train.csv")
    dataset = Dataset.from_pandas(df)

    def func(sample):
        sample["path"] = osp.join("data", sample["path"])
        return sample

    dataset = dataset.map(func)
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def load_val_dataset():
    df = pd.read_csv("data/val_denoised.csv")
    dataset = Dataset.from_pandas(df)

    def func(sample):
        sample["path"] = osp.join("data", sample["path"])
        return sample

    dataset = dataset.map(func)
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def load_test_dataset():
    df = pd.read_csv("data/test_denoised.csv")
    dataset = Dataset.from_pandas(df)

    def func(sample):
        sample["path"] = osp.join("data", sample["path"])
        return sample

    dataset = dataset.map(func)
    dataset = dataset.rename_column("path", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def create_csv(data_type, folder):
    paths = sorted(os.listdir(f"data/{data_type}/{folder}"))
    df = {
        "id": [path.split(".")[0] for path in paths],
        "path": [osp.join(f"{data_type}/{folder}", path) for path in paths],
    }
    df = pd.DataFrame(df)
    df.to_csv(f"data/{data_type}.csv", index=False)
