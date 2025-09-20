import argparse
import soundfile as sf
from transcription_pipe import TranscriptionPipe
from core.amphion_utils import load_cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json", help="config path")
    parser.add_argument("--audio_path", type=str, default="./audio_data/한국어_중국어_1.m4a", help="audio path")

    args = parser.parse_args()
    cfg = load_cfg(args.config_path)
    whisper_model_type = cfg["whisper_model_type"]
    file_name, extension = args.audio_path.split("/")[-1].split(".")

    tr_pipe = TranscriptionPipe(cfg, device_name='cpu', whisper_model_type='tiny')

    result_list = tr_pipe.run(args.audio_path)

    with open(f"audio_data/{file_name}_{whisper_model_type}.txt", "w", encoding='utf8') as f:
        f.writelines(result_list)
