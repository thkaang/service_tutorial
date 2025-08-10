import whisper


class Whisper:
    def __init__(self, type="turbo"):
        print("whisper model loading...")
        self.model = whisper.load_model(type)
        self.options = whisper.DecodingOptions()
        print("whisper model is loaded.")

    def load_audio(self, path):
        return whisper.load_audio(path)

    def decode(self, audio_segment):
        audio_segment = whisper.pad_or_trim(audio_segment)
        mel = whisper.log_mel_spectrogram(audio_segment, n_mels=self.model.dims.n_mels).to(self.model.device)
        return whisper.decode(self.model, mel, self.options).text

    def transcribe(self, path, language=None):
        result = self.model.transcribe(path, language=language)
        return result["text"]

    def transcribe_audio_data(self, samples, language=None):
        result = self.model.transcribe(samples, language=language, fp16=False)
        return result["text"]
