import os
from pydub import AudioSegment
from core.silero_vad_api import SileroVAD
from core.whisper_api import Whisper
from core.amphion_utils import *


class TranscriptionPipe:
    def __init__(self, cfg, whisper_model_type='turbo'):
        self.cfg = cfg
        self.silero_vad = SileroVAD()
        self.whisper = Whisper(type=whisper_model_type)

    def run(self, audio_path: str):
        audio = standardization(audio_path)
