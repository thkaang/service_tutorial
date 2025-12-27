import whisper


class Whisper:
    def __init__(self, type="turbo", model_path="model_files/whisper", device="cuda"):
        print("whisper model loading...")
        self.model = whisper.load_model(type, download_root=model_path, device=device)
        self.options = whisper.DecodingOptions()
        print("whisper model is loaded.")

    def load_audio(self, path):
        return whisper.load_audio(path)

    def detect_language_audio_path(self, path, idx=0):
        audio_segment = whisper.load_audio(path)
        audio_segment = whisper.pad_or_trim(audio_segment)
        mel = whisper.log_mel_spectrogram(audio_segment, n_mels=self.model.dims.n_mels).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        print(f"Detected language {idx}: {max(probs, key=probs.get)}")

    def detect_language(self, mel):
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

    def pad_or_trim(self, audio_segment):
        return whisper.pad_or_trim(audio_segment)

    def log_mel_spectrogram(self, audio_segment):
        return whisper.log_mel_spectrogram(audio_segment, n_mels=self.model.dims.n_mels).to(self.model.device)

    def decode_audio_segment(self, audio_segment):
        audio_segment = whisper.pad_or_trim(audio_segment)
        mel = whisper.log_mel_spectrogram(audio_segment, n_mels=self.model.dims.n_mels).to(self.model.device)
        return whisper.decode(self.model, mel, self.options).text

    def decode(self, mel, language):
        options = whisper.DecodingOptions(language=language)
        return whisper.decode(self.model, mel, options)

    def transcribe(self, path, language=None):
        result = self.model.transcribe(path, language=language)
        return result["text"]

    def transcribe_audio_data(self, samples, language=None):
        result = self.model.transcribe(samples, language=language, fp16=False)
        return result["text"]
