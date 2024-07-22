import os
import os.path as osp
import torch
from tqdm import tqdm

from pyannote.audio import Pipeline


def diarize(input_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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",
    )
    pipeline.to(device)

    files = sorted(os.listdir(input_dir))
    for file in tqdm(files):
        output_path = f"{output_dir}/{file.replace('.ogg', '.rttm')}"
        if osp.exists(output_path):
            continue

        rttm = pipeline(f"{input_dir}/{file}", min_speakers=0, max_speakers=2)

        with open(output_path, "w") as f:
            rttm.write_rttm(f)


if __name__ == "__main__":
    os.makedirs("data/test/diarized", exist_ok=True)
    diarize("data/test/raw", "data/test/diarized")
