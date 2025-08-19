from transcription.core.silero_vad_api import SileroVAD

audio_file = "한국어_중국어_1.m4a"
# audio_file = "한국어_중국어_1.wav"

silero_vad = SileroVAD()
speech_timestamps = silero_vad.vad(audio_file)

print(speech_timestamps)
