import os

import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from core.whisper_api import Whisper
from core.silero_vad_api import SileroVAD

# tiny < base < small < medium < turbo < large
whisper_model_type = 'turbo'
whisper = Whisper(type=whisper_model_type)
silero_vad = SileroVAD()


def slice_and_save_audio(audio_path, timestamps):
    tmp_dir = "tmp"
    file_name, extension = audio_path.split("/")[-1].split(".")

    if extension not in ["m4a", "mp3", "wav"]:
        print(f"Not allowed format: {extension}")
        return

    os.makedirs(tmp_dir, exist_ok=True)

    audio = AudioSegment.from_file(audio_path, format=extension)

    for idx, timestamp in tqdm(enumerate(timestamps, start=1)):
        start_idx = int(timestamp['start']*1000)
        end_idx = int(timestamp['end']*1000)
        sliced_audio = audio[start_idx:end_idx]
        sliced_audio.export(f"{tmp_dir}/{file_name}_{idx}.mp3")


def slice_and_transcribe_audio(audio_path, timestamps):
    target_sample_rate = 16000
    file_name, extension = audio_path.split("/")[-1].split(".")

    if extension not in ["m4a", "mp3", "wav"]:
        print(f"Not allowed format: {extension}")
        return

    audio = AudioSegment.from_file(audio_path, format=extension)
    audio = audio.set_frame_rate(16000).set_channels(1)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(np.int16).max
    script = []

    for timestamp in tqdm(timestamps):
        start_idx = int(timestamp['start']*target_sample_rate)
        end_idx = int(timestamp['end']*target_sample_rate)
        sliced_audio = samples[start_idx:end_idx]

        text = whisper.transcribe_audio_data(sliced_audio)
        script.append(text)

    with open(f"audio_data/{file_name}_{whisper_model_type}.txt", "w", encoding='utf8') as f:
        f.writelines(script)


if __name__ == '__main__':
    # audio_path = "audio_data/audio_chinese_korean.mp3"
    audio_path = "audio_data/한국어_중국어_1.m4a"
    sample_rate = 48000

    speech_timestamps = silero_vad.vad(audio_path)
    slice_and_transcribe_audio(audio_path, speech_timestamps)
