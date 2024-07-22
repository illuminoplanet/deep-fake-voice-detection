import os
import random
import numpy as np
import scipy as sp
import torch
import transformers
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    bin_totals = np.histogram(
        y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False
    )[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[: len(bin_weights)]
    prob_pred = prob_pred[: len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)

    print(f"Random seed set as {seed}")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    batch_size, num_slots = preds.shape[:2]

    background_idx = np.argmax(preds[..., 0], axis=-1)
    mask = np.ones((batch_size, num_slots), dtype=bool)
    mask[np.arange(batch_size), background_idx] = False
    preds = preds[mask].reshape(batch_size, num_slots - 1, -1)

    preds1, preds2 = preds[:, 0][:, :, None], preds[:, 1][:, None, :]
    joint_preds = (preds1 + preds2).reshape(-1, 9)
    joint_probs = sp.special.softmax(joint_preds, axis=-1)

    zero_zero = joint_probs[:, 0]
    one_zero = joint_probs[:, 1] + joint_probs[:, 3] + joint_probs[:, 4]
    zero_one = joint_probs[:, 2] + joint_probs[:, 6] + joint_probs[:, 8]
    one_one = joint_probs[:, 5] + joint_probs[:, 7]

    fake_preds = one_zero + one_one
    real_preds = zero_one + one_one

    new_preds = np.stack([fake_preds, real_preds], axis=1)
    new_labels = []
    for i in range(len(labels)):
        label = [0, 0]
        for l in labels[i]:
            if l == 0:
                continue
            label[l - 1] = 1
        new_labels.append(label)
    new_labels = np.array(new_labels)

    auc_score, ece_score, brier_score = [], [], []
    for i in range(2):
        auc_score.append(roc_auc_score(new_labels[:, i], y_score=new_preds[:, i]))
        ece_score.append(
            expected_calibration_error(new_labels[:, i], y_prob=new_preds[:, i])
        )
        brier_score.append(mean_squared_error(new_labels[:, i], y_pred=new_preds[:, i]))

    auc_score = np.mean(auc_score)
    ece_score = np.mean(ece_score)
    brier_score = np.mean(brier_score)

    total_score = 0.5 * (1 - auc_score) + 0.25 * ece_score + 0.25 * brier_score

    return {
        "auc": auc_score,
        "ece": ece_score,
        "brier": brier_score,
        "total": total_score,
    }
