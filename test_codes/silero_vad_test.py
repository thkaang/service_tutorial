from transcription.core.silero_vad_api import SileroVAD

audio_file = "../transcription/audio_data/한국어_중국어_1.m4a"

silero_vad = SileroVAD()
speech_timestamps = silero_vad.vad(audio_file)

print(speech_timestamps)
