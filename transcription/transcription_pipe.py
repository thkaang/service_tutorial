import librosa
import torch
from pyannote.audio import Pipeline
import core.separate_fast as separate_fast
from core.silero_vad_module import SileroVAD
from core.whisper_api import Whisper
from core.amphion_utils import *
from core.custom_utils import *


class TranscriptionPipe:
    def __init__(self, cfg, device_name='cuda', whisper_model_type='turbo'):
        self.cfg = cfg
        self.device_name = device_name
        self.target_sr = 16000
        self.separate_predictor = separate_fast.Predictor(args=cfg["separate"]["step1"], device=device_name)
        self.dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(torch.device(device_name))
        self.silero_vad = SileroVAD()
        self.whisper = Whisper(type=whisper_model_type, device=device_name)

    def source_separation(self, audio):
        target_sr = 44100
        mix = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"], target_sr=target_sr)
        vocals, no_vocals = self.separate_predictor.predict(mix)
        audio["sample_rate"] = target_sr
        audio["waveform"] = vocals[:, 0]

        return audio

    def run(self, audio_path: str):
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
        asr_result = asr_whisper(segment_list, audio, self.whisper)
        print("Transcription process finished")

        return asr_result

    def run_v2_wo_spkdia(self, audio_path: str):
        print("Step 1: standardization")
        audio = standardization(audio_path)
        print("Step 2: source separation")
        audio = self.source_separation(audio)

        # resample to 16kHz
        audio["waveform"] = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"], target_sr=self.target_sr)
        audio["sample_rate"] = self.target_sr

        print("Step 3: Voice Activity Detection")
        vad_list = vad_only(self.silero_vad, audio)
        segment_list = merge_over_min_length(vad_list)
        asr_result = asr_whisper(segment_list, audio, self.whisper)
        print("Transcription process finished")

        return asr_result
