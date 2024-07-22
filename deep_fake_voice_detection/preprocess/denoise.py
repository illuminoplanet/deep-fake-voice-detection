import os
import os.path as osp
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from resemble_enhance.enhancer import inference


def denoise(input_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted(os.listdir(input_dir))
    for file in tqdm(files):
        if osp.exists(f"{output_dir}/{file}"):
            continue

        audio, sr = torchaudio.load(f"{input_dir}/{file}")
        audio = audio.mean(dim=0)

        audio, sr = inference.denoise(audio, sr, device)
        audio, sr = inference.enhance(
            audio, sr, device, nfe=64, solver="midpoint", lambd=0.1, tau=0.5
        )
        sf.write(f"{output_dir}/{file}", audio.reshape(-1).cpu().numpy(), sr)


if __name__ == "__main__":
    os.makedirs("data/test/denoised", exist_ok=True)
    denoise("data/test/raw", "data/test/denoised")
