from dataclasses import dataclass

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy as sp
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor

from deep_fake_voice_detection.datasets.dataset import load_test_dataset
from deep_fake_voice_detection.datasets.augmentation import (
    prepare_dataset,
    AudioAugmenter,
)
from deep_fake_voice_detection.models.model import WhisperWithSlotAttention


def load_model(model_path):
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=3,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_path,
        do_normalize=True,
    )
    model = WhisperWithSlotAttention.from_pretrained(
        model_path,
        config=config,
    )
    return model, feature_extractor


@dataclass
class DataCollator:
    def __call__(self, features):
        input_features = torch.stack(
            [feature["input_features"] for feature in features]
        )
        batch = {"input_features": input_features}
        return batch


def predict(
    dataset,
    model,
    batch_size=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollator(),
        num_workers=0,
    )

    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):

            input_features = batch["input_features"].to(device)
            outputs = model(input_features)["logits"].float()

            preds = torch.softmax(outputs, dim=-1).cpu().numpy()
            all_predictions.append(preds)

    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions


if __name__ == "__main__":
    model_path = "model/checkpoint-11550"
    model, feature_extractor = load_model(model_path=model_path)

    augmenter = AudioAugmenter(sample_rate=16000)
    dataset = load_test_dataset()
    dataset.set_transform(
        lambda sample: prepare_dataset(sample, feature_extractor, None, mixup=False)
    )

    preds = predict(dataset, model)
    batch_size, num_slots = preds.shape[:2]

    background_idx = np.argmax(preds[..., 0], axis=-1)
    mask = np.ones((batch_size, num_slots), dtype=bool)
    mask[np.arange(batch_size), background_idx] = False
    preds = preds[mask].reshape(batch_size, num_slots - 1, -1)

    preds1, preds2 = preds[:, 0][:, :, None], preds[:, 1][:, None, :]

    inv_temperature = 1
    joint_preds = (preds1 + preds2).reshape(-1, 9)
    joint_probs = sp.special.softmax(joint_preds * inv_temperature, axis=-1)

    zero_zero = joint_probs[:, 0]
    one_zero = joint_probs[:, 1] + joint_probs[:, 3] + joint_probs[:, 4]
    zero_one = joint_probs[:, 2] + joint_probs[:, 6] + joint_probs[:, 8]
    one_one = joint_probs[:, 5] + joint_probs[:, 7]

    fake_preds = one_zero + one_one
    real_preds = zero_one + one_one

    df = pd.read_csv("data/sample_submission.csv")
    df["fake"] = fake_preds.tolist()
    df["real"] = real_preds.tolist()

    activity = pd.read_csv("data/test_activity.csv")
    df.loc[activity["active"] == 0, ["real", "fake"]] = 0

    df.to_csv("submission.csv", index=False)
