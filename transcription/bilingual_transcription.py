import os

import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from core.whisper_api import Whisper
from core.silero_vad_api import SileroVAD

# tiny < base < small < medium < turbo < large
whisper_model_type = 'turbo'
# whisper_model_type = 'small'
whisper = Whisper(type=whisper_model_type)
silero_vad = SileroVAD()


def slice_audio(audio, timestamps, min_sliced_audio_sec=3):
    min_sliced_audio_len = min_sliced_audio_sec*1000
    start_idx = None
    sliced_audio_list = []

    for idx, timestamp in enumerate(tqdm(timestamps), start=1):
        if start_idx is None:
            start_idx = int(timestamp['start']*1000)
        end_idx = int(timestamp['end']*1000)
        if end_idx - start_idx >= min_sliced_audio_len:  # under 3 sec
            # when the next start timestamp ~ last end timestamp is under 3 sec
            if (idx < len(timestamps) and int(timestamps[-1]['end']*1000)
                    - int(timestamps[idx]['start']*1000) < min_sliced_audio_len):
                end_idx = timestamps[-1]['end']*1000
            sliced_audio = audio[start_idx:end_idx]
            sliced_audio_list.append(sliced_audio)
            start_idx = None
    return sliced_audio_list


def slice_and_save_audio(audio_path, timestamps, threshold=0.5, detect_language=False):
    tmp_dir = f"tmp/thresold_{threshold}"
    file_name, extension = audio_path.split("/")[-1].split(".")

    if extension not in ["m4a", "mp3", "wav"]:
        print(f"Not allowed format: {extension}")
        return

    audio = AudioSegment.from_file(audio_path, format=extension)
    os.makedirs(tmp_dir, exist_ok=True)

    sliced_audio_list = slice_audio(audio, timestamps)
    for idx, sliced_audio in enumerate(tqdm(sliced_audio_list), start=1):
        sliced_file_path = f"{tmp_dir}/{file_name}_{idx}.mp3"
        sliced_audio.export(sliced_file_path)

        if detect_language:
            whisper.detect_language(sliced_file_path, idx=idx)


def slice_and_transcribe_audio(audio_path, timestamps, min_sliced_audio_sec=3):
    target_sample_rate = 16000
    file_name, extension = audio_path.split("/")[-1].split(".")

    if extension not in ["m4a", "mp3", "wav"]:
        print(f"Not allowed format: {extension}")
        return

    audio = AudioSegment.from_file(audio_path, format=extension)
    sliced_audio_list = slice_audio(audio, timestamps, min_sliced_audio_sec=min_sliced_audio_sec)
    script = []

    for idx, sliced_audio in enumerate(tqdm(sliced_audio_list), start=1):
        sliced_audio = sliced_audio.set_frame_rate(target_sample_rate).set_channels(1)
        samples = np.array(sliced_audio.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(np.int16).max
        text = whisper.transcribe_audio_data(samples)
        print(f"idx: {idx}, text: {text}")
        script.append(text)

    with open(f"audio_data/{file_name}_{whisper_model_type}.txt", "w", encoding='utf8') as f:
        f.writelines(script)


if __name__ == '__main__':
    # audio_path = "audio_data/audio_chinese_korean.mp3"
    audio_path = "audio_data/한국어_중국어_1.m4a"
    # audio_path = "audio_data/한국어_영어_1.m4a"
    threshold = 0.5
    min_sliced_audio_sec = 3

    speech_timestamps = silero_vad.vad(audio_path, threshold=threshold)
    # slice_and_save_audio(audio_path, speech_timestamps, threshold=threshold, detect_language=False)
    slice_and_transcribe_audio(audio_path, speech_timestamps)
