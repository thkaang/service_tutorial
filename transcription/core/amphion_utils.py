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
