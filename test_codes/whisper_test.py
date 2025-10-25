import whisper
import os
import time

print(os.getcwd())
path = os.path.abspath("../transcription/audio_data/한국어_중국어_1.m4a")
print(path)
model = whisper.load_model("turbo")
# model = whisper.load_model("medium")
# model = whisper.load_model("small")
# model = whisper.load_model("base")
# model = whisper.load_model("tiny")

start = time.time()
result = model.transcribe(path, language=None)
end = time.time()
print(result["text"])
print(f'{end - start:.5f} sec')
print(result)
