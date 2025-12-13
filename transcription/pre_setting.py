import os
import torch

def predownload_silero_vad(
    hub_dir: str = "model_files/torchhub",
    repo: str = "snakers4/silero-vad",
    model: str = "silero_vad",
):
    # 1) torch hub 캐시 디렉토리 고정 (컨테이너 볼륨 마운트 권장)
    os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)

    # 2) Silero VAD 다운로드/로딩 (캐시에 없으면 받음, 있으면 재사용)
    print(f"[INFO] torch.hub dir = {hub_dir}")
    print(f"[INFO] loading {repo}:{model} ...")

    vad_model, utils = torch.hub.load(
        repo_or_dir=repo,
        model=model,
        force_reload=False,   # 캐시 있으면 재다운로드 X
        trust_repo=True,      # torch 2.x 계열에서 요구될 수 있음
    )

    # 3) 간단 체크 (정상 로드 여부)
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    print("[OK] Silero VAD loaded and cached successfully.")
    print(f"[INFO] hub cache dir content: {hub_dir}")

    return vad_model, utils


if __name__ == "__main__":
    hub_dir = os.environ.get("TORCH_HUB_DIR", "model_files/torchhub")
    predownload_silero_vad(hub_dir=hub_dir)