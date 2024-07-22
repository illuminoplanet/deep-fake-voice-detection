import numpy as np
import torch
import audiomentations as A
from audiomentations.core.transforms_interface import BaseWaveformTransform
from numpy.typing import NDArray


class RandomReduce(BaseWaveformTransform):
    def __init__(
        self,
        min_duration_samples: int = None,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.min_duration_samples = min_duration_samples

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        sample_length = samples.shape[-1]
        if sample_length <= self.min_duration_samples:
            return samples
        reduce_length = np.random.randint(self.min_duration_samples, sample_length + 1)
        reduce_left = np.random.randint(0, sample_length - reduce_length + 1)
        return samples[..., reduce_left : reduce_left + reduce_length]


class RandomExtend(BaseWaveformTransform):
    def __init__(
        self,
        max_duration_samples: int = None,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.max_duration_samples = max_duration_samples

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        sample_length = samples.shape[-1]
        if sample_length >= self.max_duration_samples:
            return samples

        extend_length = np.random.randint(
            0, self.max_duration_samples - sample_length + 1
        )
        extend_left = np.random.randint(0, extend_length + 1)
        if samples.ndim == 1:
            extend_width = (extend_left, extend_length - extend_left)
        else:
            extend_width = ((0, 0), (extend_left, extend_length - extend_left))
        extend_mode = np.random.choice(["wrap", "reflect"])

        return np.pad(samples, extend_width, mode=extend_mode)


class RandomAdjustDuration(BaseWaveformTransform):
    def __init__(
        self,
        duration_samples: int = None,
        duration_seconds: float = None,
        p: float = 0.5,
    ):
        super().__init__(p)
        assert duration_samples is not None or duration_seconds is not None
        if duration_samples is not None and duration_seconds is not None:
            raise ValueError(
                "should have duration_samples or duration_seconds, but not both"
            )
        elif duration_seconds:
            assert duration_seconds > 0
            self.get_target_samples = lambda sr: int(duration_seconds * sr)
        elif duration_samples:
            assert duration_samples > 0
            self.get_target_samples = lambda sr: duration_samples

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        target_samples = self.get_target_samples(sample_rate)
        sample_length = samples.shape[-1]

        if sample_length == target_samples:
            return samples

        elif sample_length > target_samples:
            start = np.random.randint(0, sample_length - target_samples)
            return samples[..., start : start + target_samples]

        elif sample_length < target_samples:
            padding_length = target_samples - sample_length
            padding_left = np.random.randint(0, padding_length)
            if samples.ndim == 1:
                pad_width = (padding_left, padding_length - padding_left)
            else:
                pad_width = ((0, 0), (padding_left, padding_length - padding_left))
            return np.pad(samples, pad_width, mode="constant")


class AudioAugmenter:
    def __init__(self, sample_rate):
        self.preprocess = A.Compose(
            [
                A.Trim(top_db=20, p=1.0),
                A.OneOf(
                    [
                        RandomReduce(min_duration_samples=15000, p=1.0),
                        RandomExtend(max_duration_samples=65000, p=1.0),
                    ],
                    p=0.5,
                ),
                RandomAdjustDuration(duration_samples=80000, p=1.0),
            ],
            p=1.0,
        )
        self.wave_augment = A.Compose(
            [
                A.AddColorNoise(),
                A.PitchShift(min_semitones=-2, max_semitones=3),
                A.TimeStretch(min_rate=0.8, max_rate=1.2),
                A.BitCrush(min_bit_depth=16, max_bit_depth=32),
                A.TimeMask(max_band_part=0.2),
                A.Normalize(p=1.0),
            ]
        )
        self.spec_augment = A.SpecFrequencyMask(max_mask_fraction=0.2)

        self.sample_rate = sample_rate

    def __call__(self, audios, labels=None, mode="augment"):
        if mode == "mixup":
            assert labels is not None

            augmented_audios, augmented_labels = self._mixup(audios, labels)
            return augmented_audios, augmented_labels
        else:
            func = {
                "preprocess": self.preprocess,
                "wave_augment": self.wave_augment,
                "spec_augment": self.spec_augment,
            }[mode]

            if mode == "spec_augment":
                return np.array([func(audio) for audio in audios])
            else:
                augmented_audios = np.array(
                    [func(audio, sample_rate=self.sample_rate) for audio in audios]
                )
            return augmented_audios

    def _mixup(self, audios, labels):
        empty_audios = np.zeros_like(audios)
        empty_labels = [0] * len(audios)

        audios = np.concatenate([audios, empty_audios], axis=0)
        labels = labels + empty_labels

        indices = np.random.permutation(len(audios))
        mixup_audios = []
        mixup_labels = []
        for i, index in enumerate(indices[: len(audios) - len(empty_audios)]):
            mixup_ratio = np.random.beta(0.5, 0.5)
            mixup_audio = mixup_ratio * audios[i] + (1 - mixup_ratio) * audios[index]
            mixup_label = np.array([0, labels[i], labels[index]])
            mixup_audios.append(mixup_audio)
            mixup_labels.append(mixup_label)

        return mixup_audios, mixup_labels

    def _augment(self, samples, sample_rate):
        samples = self.wave_augment(samples=samples, sample_rate=sample_rate)
        for transform in self.spec_augment:
            samples = transform(samples=samples, sample_rate=sample_rate)
        return samples


def prepare_dataset(samples, feature_extractor, augmenter=None, mixup=True):
    audios = [sample["array"] for sample in samples["audio"]]
    labels = None

    if "label" in samples:
        labels = samples["label"]
        if isinstance(samples["label"][0], str):
            labels = [sample.split(",") for sample in samples["label"]]
            labels = [np.array(list(map(int, label))) for label in labels]

    if augmenter is not None:
        audios = augmenter(audios, mode="preprocess")
        if mixup and "label" in samples:
            audios, labels = augmenter(audios, labels, mode="mixup")
        audios = augmenter(audios, mode="wave_augment")

    samples["input_features"] = feature_extractor(
        audios,
        sampling_rate=16000,
        do_normalize=True,
    ).input_features

    if augmenter is not None:
        samples["input_features"] = augmenter(
            samples["input_features"], mode="spec_augment"
        )

    samples["input_features"] = torch.tensor(samples["input_features"])
    if labels:
        samples["labels"] = torch.tensor(labels)
    return samples
