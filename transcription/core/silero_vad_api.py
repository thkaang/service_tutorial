import torch
import numpy as np
from pydub import AudioSegment


class SileroVAD:
    def __init__(self):
        self.model, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad')

        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        print("Silero VAD is loaded")

    def __m4a_to_wav(self, audio_file):
        audio = AudioSegment.from_file(audio_file, format="m4a")
        audio = audio.set_channels(1).set_frame_rate(16000)

        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        return torch.from_numpy(samples)

    def vad(self, audio_file, threshold=0.5):
        format = audio_file.split(".")[-1]
        if format == "m4a":
            audio_data = self.__m4a_to_wav(audio_file)
        elif format in ["wav", "mp3"]:
            audio_data = self.read_audio(audio_file)
        else:
            print(f"This audio format isn't supported for VAD: {format}")
            return None

        speech_timestamps = self.get_speech_timestamps(
            audio_data,
            self.model,
            threshold=threshold,
            return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )

        return speech_timestamps
