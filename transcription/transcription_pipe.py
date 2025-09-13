import librosa
import core.separate_fast as separate_fast
from core.silero_vad_api import SileroVAD
from core.whisper_api import Whisper
from core.amphion_utils import *


class TranscriptionPipe:
    def __init__(self, cfg, device_name='cuda', whisper_model_type='turbo'):
        self.cfg = cfg
        self.separate_predictor = separate_fast.Predictor(args=cfg["separate"]["step1"], device=device_name)
        self.silero_vad = SileroVAD()
        self.whisper = Whisper(type=whisper_model_type)

    def source_separation(self, audio):
        target_sr = 44100
        mix = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"], target_sr=target_sr)
        vocals, no_vocals = self.separate_predictor.predict(mix)
        audio["sample_rate"] = target_sr
        audio["waveform"] = vocals[:, 0]

        return audio

    def run(self, audio_path: str):
        audio = standardization(audio_path)
        audio = self.source_separation(audio)
        return audio
