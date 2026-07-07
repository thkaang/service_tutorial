import io
import sys
import os
import argparse
from flask import Flask, jsonify, request, make_response
from transcription_pipe import TranscriptionPipe
from core.amphion_utils import load_cfg
from core.custom_utils import allowed_file

ALLOWED_EXTENSIONS = ["wav", "m4a", "mp3"]
app = Flask(__name__,
            static_folder="server/dist/public",  # 빌드된 React
            static_url_path="")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json", help="config path")
    known_args = [arg for arg in sys.argv if arg.startswith('--config_path')]
    return parser.parse_args(known_args)


config_path = os.environ.get("CONFIG_PATH", None)
args = get_args()
config_path = config_path if config_path is not None else args.config_path
cfg = load_cfg(config_path)
whisper_model_type = cfg["whisper_model_type"]
os.environ["HF_TOKEN"] = cfg["huggingface_token"]
tr_pipe = TranscriptionPipe(cfg, device_name='cuda', whisper_model_type=whisper_model_type)


# 여기에 /api 라우트 구현
@app.route("/api/ping")
def ping():
    return jsonify({"message": "pong from Flask"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files.get("file")
    language1 = request.form.get("language1", "ko")
    language2 = request.form.get("language2", "en")

    allowed, audio_format = allowed_file(file.filename, ALLOWED_EXTENSIONS)

    if file and allowed:
        audio_buffer = io.BytesIO(file.read())
        restrict_lang_dict = {
            "1st_lang": language1,
            "2nd_lang": language2
        }
        transcription_list = tr_pipe.run(audio_buffer, audio_format, restrict_lang_dict=restrict_lang_dict)
        transcription = '\n'.join(transcription_list)

        # text/plain으로 반환
        response = make_response(transcription, 200)
        response.headers["Content-Type"] = "text/plain; charset=utf-8"

        # CORS 허용
        response.headers["Access-Control-Allow-Origin"] = "*"

        return response

    # 허용되지 않은 확장자
    return make_response("File type not allowed. Please upload wav/m4a/mp3.", 400)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
