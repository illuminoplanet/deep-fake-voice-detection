import warnings

warnings.filterwarnings("ignore")
import argparse
from dataclasses import dataclass

import wandb
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoFeatureExtractor,
)

from deep_fake_voice_detection.datasets.dataset import (
    load_train_dataset,
    load_val_dataset,
)
from deep_fake_voice_detection.datasets.augmentation import (
    prepare_dataset,
    AudioAugmenter,
)
from deep_fake_voice_detection.utils import seed_everything, compute_metrics
from deep_fake_voice_detection.models.model import WhisperWithSlotAttention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


@dataclass
class DataCollator:
    def __call__(self, features):
        batch = {}
        batch["input_features"] = torch.stack(
            [feature["input_features"] for feature in features]
        )
        if "labels" in features[0]:
            batch["labels"] = torch.stack(
                [feature["labels"] for feature in features], dim=0
            )
        return batch


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
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=config,
    )
    return model, feature_extractor


def load_dataset():
    train_dataset = load_train_dataset()
    val_dataset = load_val_dataset()
    dataset = {"train": train_dataset, "test": val_dataset}

    augmenter = AudioAugmenter(sample_rate=16000)
    dataset["train"].set_transform(
        lambda sample: prepare_dataset(sample, feature_extractor, augmenter, mixup=True)
    )
    dataset["test"].set_transform(
        lambda sample: prepare_dataset(sample, feature_extractor, None, mixup=False)
    )
    data_collator = DataCollator()
    return dataset, data_collator


def load_trainer(model, feature_extractor, dataset, data_collator, args):
    training_args = TrainingArguments(
        output_dir="model",
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        eval_steps=400,
        save_steps=800,
        logging_steps=10,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="total",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else "none",
    )
    trainer = Trainer(
        model=model,
        tokenizer=feature_extractor,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        args=training_args,
    )
    return trainer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model_path = "openai/whisper-small.en"
    model, feature_extractor = load_model(model_path=model_path)

    dataset, data_collator = load_dataset()
    if args.wandb:
        wandb.init(
            project="deep-fake-voice-detection",
            name="whisper-small",
        )

    trainer = load_trainer(model, feature_extractor, dataset, data_collator, args)
    trainer.train()
    trainer.save_model("model")
