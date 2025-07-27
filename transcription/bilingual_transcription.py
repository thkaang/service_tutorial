import soundfile as sf
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from core.whisper_api import Whisper
from core.silero_vad_api import SileroVAD

whisper = Whisper(type='tiny')
silero_vad = SileroVAD()


if __name__ == '__main__':
    audio_path = "../audio_chinese_korean.mp3"
    sample_rate = 48000
    audio_test = AudioSegment.from_mp3(audio_path)
    samples = np.array(audio_test.get_array_of_samples())
    sample_rate_test = audio_test.frame_rate
    channels = audio_test.channels
    audio = whisper.load_audio(audio_path)
    speech_timestamps = silero_vad.vad(audio_path)

    for timestamp in tqdm(speech_timestamps):
        start_idx = int(timestamp['start']*16000)
        end_idx = int(timestamp['end']*16000)
        audio_segment = audio[start_idx:end_idx]
        # audio_segment_test = audio_test[]

        text = whisper.decode(audio_segment)
        print(text)
