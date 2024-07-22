import os
from typing import List, Tuple
import pandas as pd


def get_sorted_rttm_paths(directory: str) -> List[str]:
    rttm_files = [f for f in os.listdir(directory) if f.endswith(".rttm")]
    return sorted([os.path.join(directory, f) for f in rttm_files])


def read_rttm_file(file_path: str) -> List[Tuple[float, float]]:
    segments = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4 or parts[0] != "SPEAKER":
                continue
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            segments.append((start_time, end_time))
    return segments


def find_speaker_bounds(
    segments: List[Tuple[float, float]]
) -> Tuple[float, float, float]:
    if not segments:
        return None
    start_times, end_times = zip(*segments)
    start = min(start_times)
    end = max(end_times)
    return end - start


def process_rttm_files(file_paths: List[str]) -> pd.DataFrame:
    results = []
    for file_path in file_paths:
        segments = read_rttm_file(file_path)
        duration = find_speaker_bounds(segments)
        if duration:
            results.append(
                {
                    "id": os.path.basename(file_path).split(".")[0],
                    "active": duration > 0.5,
                }
            )
        else:
            results.append(
                {
                    "id": os.path.basename(file_path).split(".")[0],
                    "active": False,
                }
            )
    return pd.DataFrame(results)


if __name__ == "__main__":
    output_csv_path = "data/test_activity.csv"
    directory = "data/test/diarized"

    rttm_files = get_sorted_rttm_paths(directory)
    df = process_rttm_files(rttm_files)
    activity_df = pd.read_csv("data/test_activity.csv")
    df.to_csv(output_csv_path, index=False)
