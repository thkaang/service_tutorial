import time
import whisperx

device = "cpu"
compute_type = "int8"
audio_file = "audio_korean.mp3"

model = whisperx.load_model("large-v3", device, compute_type=compute_type)
start = time.time()
result = model.transcribe(audio_file)
end = time.time()
print(f'{end - start:.5f} sec')
print(result["segments"])
