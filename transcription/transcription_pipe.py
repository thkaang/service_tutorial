import time
import librosa
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pyannote.audio import Pipeline
import core.separate_fast as separate_fast
from core.uvr.separate import _audio_pre_
from core.silero_vad_module import SileroVAD
from core.whisper_api import Whisper
from core.amphion_utils import *
from core.custom_utils import *


class TranscriptionPipe:
    def __init__(self, cfg, device_name='cuda', whisper_model_type='turbo'):
        self.cfg = cfg
        self.device_name = device_name
        self.target_sr = 16000
        self.separate_predictor = None
        if device_name == 'cuda':
            self.separate_predictor = _audio_pre_(model_path=cfg['separate_gpu']['model_path'], device=device_name, is_half=True)
        else:
            self.separate_predictor = separate_fast.Predictor(args=cfg["separate"]["step1"], device=device_name)
        self.dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(torch.device(device_name))
        self.silero_vad = SileroVAD()
        self.whisper = Whisper(type=whisper_model_type, device=device_name)

    def source_separation(self, audio):
        target_sr = 44100
        vocals, no_vocals = self.separate_predictor.predict(audio, target_sr)
        audio["sample_rate"] = target_sr
        audio["waveform"] = vocals[:, 0]

        return audio

    def get_audio_chunks(self, audio):
        audio_chunks = []
        resampled_audio = {}
        audio_length = len(audio["waveform"]) / audio["sample_rate"]

        if audio_length > self.cfg["vad"]["preprocess"]["MAX_SEGMENT_LENGTH"]:  # for long audio
            print("Step 1.1: Voice Activity Detection (for long audio)")
            print("Step 1.1.1: resampling for pre vad")
            resampled_audio["waveform"] = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"],
                                                           target_sr=self.target_sr)
            resampled_audio["sample_rate"] = self.target_sr
            print("Step 1.1.2: execute pre vad")
            pre_vad_list = vad_only(self.silero_vad, resampled_audio)
            pre_segment_list = merge_over_min_length(pre_vad_list, self.cfg["vad"]["preprocess"], True)
            for pre_timestamp in pre_segment_list:
                start_frame = int(pre_timestamp["start"] * audio["sample_rate"])
                end_frame = int(pre_timestamp["end"] * audio["sample_rate"])
                audio_chunks.append({
                    "waveform": audio["waveform"][start_frame:end_frame],
                    "name": audio["name"],
                    "sample_rate": audio["sample_rate"]
                })
        else:
            audio_chunks = [audio]

        return audio_chunks

    def run(self, audio_path: str):
        start = time.time()
        print("Step 1: standardization")
        audio = standardization(audio_path)

        print("Step 2: source separation")
        audio = self.source_separation(audio)

        # resample to 16kHz
        audio["waveform"] = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"], target_sr=self.target_sr)
        audio["sample_rate"] = self.target_sr

        print("Step 3: speaker diarization")
        speaker_info_df = speaker_diarization(audio, self.dia_pipeline, device_name=self.device_name)
        print("Step 4: Voice Activity Detection")
        vad_list = self.silero_vad.vad(speaker_info_df, audio)
        segment_list = cut_by_speaker_label(vad_list)
        print("Step 5: Transcription")
        asr_result = asr_whisper(segment_list, audio, self.whisper)
        print("Transcription process finished")
        elapsed_time = time.time() - start
        audio_length = len(audio["waveform"]) / audio["sample_rate"]
        print(f"Total elapsed time: {elapsed_time:.3f} sec")
        print(f"xRT: {elapsed_time / audio_length:.3f} xRT")

        return asr_result

    def run_v2_wo_spkdia(self, audio_path: str):
        start = time.time()
        asr_results = []
        print("Step 1: standardization")
        audio = standardization(audio_path)

        audio_chunks = self.get_audio_chunks(audio)

        for idx, audio_chunk in enumerate(tqdm(audio_chunks), start=1):
            # sf.write(f"./audio_data/test_{audio_chunk['name']}_{idx}.wav", audio_chunk["waveform"], samplerate=audio_chunk["sample_rate"])
            print(f"Step 2_({idx}): source separation")
            audio_chunk = self.source_separation(audio_chunk)

            # resample to 16kHz
            print(f"Step 2.1_({idx}): resampling for vad")
            audio_chunk["waveform"] = librosa.resample(audio_chunk["waveform"], orig_sr=audio_chunk["sample_rate"], target_sr=self.target_sr)
            audio_chunk["sample_rate"] = self.target_sr

            print(f"Step 3_({idx}): Voice Activity Detection")
            vad_list = vad_only(self.silero_vad, audio_chunk)
            segment_list = merge_over_min_length(vad_list, self.cfg['vad']['transcription'])
            print(f"Step 4_({idx}): Transcription")
            asr_result = asr_whisper(segment_list, audio_chunk, self.whisper)
            asr_result = f"{idx}. {' '.join(asr_result)}\n"
            asr_results.append(asr_result)

        print("Transcription process finished")
        elapsed_time = time.time() - start
        audio_length = len(audio["waveform"]) / audio["sample_rate"]
        print(f"Total elapsed time: {elapsed_time:.3f} sec")
        print(f"xRT: {elapsed_time / audio_length:.3f} xRT")

        return asr_results
