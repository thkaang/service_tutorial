# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import torch
import numpy as np
import pandas as pd
from pydub import AudioSegment


def standardization(audio_path: str):
    name = os.path.basename(audio_path)
    audio = AudioSegment.from_file(audio_path)

    # Convert the audio file to WAV format
    # audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    sample_rate = audio.frame_rate
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)  # Set to mono

    print(f"[LOG] Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    print(f"[LOG] Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize

    print(f"[LOG] waveform shape: {waveform.shape}")
    print(f"[LOG] waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": sample_rate,
    }


def speaker_diarization(audio, dia_pipeline, device_name='cuda'):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """
    print(f"[LOG] Start speaker diarization")
    print(f"[LOG] audio waveform shape: {audio['waveform'].shape}")

    device = torch.device(device_name)

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        }
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    print(f"[LOG] diarize_df: {diarize_df}")

    return diarize_df


def cut_by_speaker_label(vad_list):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 2  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            print(
                f"cut_by_speaker_label > segment longer than 30s, force trimming to 30s smaller segments"
            )
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH  # update end time
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start  # update start time
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    print(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    print(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list


def load_cfg(cfg_path):
    """
    Load configuration from a JSON file.

    Args:
        cfg_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found. Please copy, configure, and rename `config.json.example` to `{cfg_path}`."
        )
    with open(cfg_path, "r") as f:
        try:
            cfg = json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise TypeError(
                "Please finish the `// TODO:` in the `config.json` file before running the script. Check README.md for details."
            )
    return cfg
