import whisper
import os
import time


class Whisper:
    def __init__(self, type="turbo"):
        print("whisper model loading...")
        self.model = whisper.load_model(type)
        print("whisper model is loaded.")

    def transcribe(self, path, language=None):
        result = self.model.transcribe(path, language=language)
        return result["text"]
