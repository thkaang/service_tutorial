import argparse
import io
import os
import time
from transcription_pipe import TranscriptionPipe
from core.amphion_utils import load_cfg
from core.custom_utils import allowed_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json", help="config path")
    parser.add_argument("--audio_path", type=str, default="./audio_data/한국어_중국어_1.m4a", help="audio path")

    args = parser.parse_args()
    cfg = load_cfg(args.config_path)
    whisper_model_type = cfg["whisper_model_type"]
    tr_pipe_type = cfg["transcription_pipe_type"]
    os.environ["HF_TOKEN"] = cfg["huggingface_token"]
    allowed_extensions = ["wav", "m4a", "mp3"]
    allowed, extension = allowed_file(args.audio_path, allowed_extensions)
    file_name = args.audio_path.split("/")[-1].split(".")[0]

    tr_pipe = TranscriptionPipe(cfg, device_name='cuda', whisper_model_type=whisper_model_type)
    if allowed:
        with open(args.audio_path, "rb") as af:
            audio_bytes = af.read()
            audio_buffer = io.BytesIO(audio_bytes)
            start = time.time()
            if tr_pipe_type == "v1":
                result_list = tr_pipe.run(audio_buffer, extension)
            else:
                result_list = tr_pipe.run_v2_wo_spkdia(audio_buffer, extension)
            print(f"Total elapsed time: {(time.time() - start):.3f} sec")

            with open(f"audio_data/{file_name}_{whisper_model_type}_{tr_pipe_type}.txt", "w", encoding='utf8') as f:
                f.writelines(result_list)
    else:
        print(f"{extension} format is not allowed.")
