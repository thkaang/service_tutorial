import os
import numpy as np
import librosa
import soundfile as sf
import torch
from pyannote.audio import Pipeline

# ------------------------------------
# 설정
# ------------------------------------
AUDIO_PATH = "../transcription/audio_data/한국어_중국어_1.m4a"  # M4A 입력
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

# pyannote 파이프라인 로드
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN,
)

# ------------------------------------
# 1) M4A 로드 (모노, 원래 sr 유지)
# ------------------------------------
wave_orig, sr_orig = librosa.load(AUDIO_PATH, sr=None, mono=True)  # float32 (T,)

# ------------------------------------
# 2) diarization 입력용 파형 준비
#    - sr > 16000 → 16k로 다운샘플
#    - sr <= 16000 → 그대로 사용
# ------------------------------------
sr_for_diar = sr_orig
wave_for_diar = wave_orig
if sr_orig > 16000:
    wave_for_diar = librosa.resample(
        wave_orig, orig_sr=sr_orig, target_sr=16000, res_type="kaiser_best"
    )
    sr_for_diar = 16000

# >>> 핵심 수정: numpy -> torch.FloatTensor, (C, T) 형태로 변환
wave_tensor = torch.from_numpy(wave_for_diar).to(torch.float32).unsqueeze(0)  # (1, T)

# ------------------------------------
# 3) diarization 실행
# ------------------------------------
diarization = pipeline({"waveform": wave_tensor, "sample_rate": sr_for_diar})

print("time stamp")
print(diarization)

# ------------------------------------
# 4) 화자별 segment 잘라서 저장 (원본 sr 기준)
# ------------------------------------
speaker_chunks = {}
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_sample = int(max(0, np.floor(turn.start * sr_orig)))
    end_sample   = int(min(len(wave_orig), np.ceil(turn.end * sr_orig)))
    seg = wave_orig[start_sample:end_sample]
    if seg.size == 0:
        continue
    speaker_chunks.setdefault(speaker, []).append(seg)

# 화자별로 하나의 파일로 합쳐 저장
os.makedirs("speakers", exist_ok=True)
for spk, segs in speaker_chunks.items():
    merged = np.concatenate(segs, axis=0) if len(segs) > 1 else segs[0]
    out_path = os.path.join("speakers", f"{spk}.wav")
    sf.write(out_path, merged, sr_orig)
    print(f"Saved: {out_path} (sr={sr_orig}, {len(merged)/sr_orig:.2f}s)")
