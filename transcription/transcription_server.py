import argparse
import os
import sys
from flask import Flask, request, render_template, jsonify
from core.whisper_api import Whisper


app = Flask(__name__, static_url_path= '/static')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ui_port', default=5001)
    known_args = [arg for arg in sys.argv if arg.startswith('--ui_port')]
    return parser.parse_args(known_args)


class STT:
    def __init__(self):
        self.stt = Whisper()

    def transcribe(self, path):
        return self.stt.transcribe(path)


ui_port = os.environ.get("UI_PORT", None)


args = get_args()
ui_port = ui_port if ui_port is not None else args.ui_port
upload_folder = "speech_wav"
os.makedirs(upload_folder, exist_ok=True)
stt = STT()


@app.route("/")
def rendering_template():
    return render_template("main.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    print("file is uploaded")
    if 'audio' not in request.files:
        return "No audio file part", 400
    file = request.files['audio']
    if file.filename == '':
        return "No selected file", 400
    save_path = os.path.join(upload_folder, file.filename)
    file.save(save_path)

    text = stt.transcribe(save_path)

    print(f"File {file.filename} uploaded successfully!")
    print(f"script: {text}")

    return jsonify({
        "text": text
    })


if __name__ == '__main__':
    print("running")
    app.run(host="0.0.0.0", port=ui_port)
