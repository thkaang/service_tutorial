import io
import string
import sys
import os
import argparse
import requests
import secrets
from flask import Flask, send_from_directory, jsonify, request, make_response
from transcription_pipe import TranscriptionPipe
from core.amphion_utils import load_cfg
from core.custom_utils import allowed_file

ALLOWED_EXTENSIONS = ["wav", "m4a", "mp3"]
app = Flask(__name__,
            static_folder="server/dist/public",  # 빌드된 React
            static_url_path="")


def secure_random_string_16():
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(16))


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
SECRET_KEY = secure_random_string_16()
SECRET_ENDPOINT = "http://127.0.0.1:5000/transcribe"

tr_pipe = TranscriptionPipe(cfg, device_name='cuda', whisper_model_type=whisper_model_type)


# React 정적 파일 서빙
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    # JS/CSS/이미지 등
    file_path = os.path.join(app.static_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    # 그 외 라우트는 React에 맡김 (SPA)
    return send_from_directory(app.static_folder, "index.html")


# 여기에 /api 라우트 구현
@app.route("/api/ping")
def ping():
    return jsonify({"message": "pong from Flask"})


@app.route("/transcribe_request", methods=["post"])
def transcribe_request():
    # 파일이 있는지 확인
    if "file" not in request.files:
        return make_response("No file part in the request", 400)

    file = request.files["file"]

    if file is None or file.filename == "":
        return make_response("No selected file", 400)

    files = {
        "file": (file.filename, file.stream, file.content_type)
    }

    data = {
        "secret_key": SECRET_KEY
    }

    response = requests.post(SECRET_ENDPOINT, files=files, data=data)
    response = make_response(response.content, response.status_code)
    response.headers["Content-Type"] = response.headers.get(
        "Content-Type",
        "text/plain; charset=utf-8",
    )
    return response


@app.route("/transcribe", methods=["POST"])
def transcribe():
    secret_key = request.form.get("secret_key")
    file = request.files.get("file")

    if secret_key != SECRET_KEY:
        return jsonify({"error": "invalid key"}), 403

    allowed, audio_format = allowed_file(file.filename, ALLOWED_EXTENSIONS)

    if file and allowed:
        audio_buffer = io.BytesIO(file.read())

        transcription_list = tr_pipe.run(audio_buffer, audio_format)
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
