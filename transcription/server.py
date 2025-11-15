# app.py
from flask import Flask, send_from_directory, jsonify, request, make_response
from werkzeug.utils import secure_filename
import os

# 허용 확장자
ALLOWED_EXTENSIONS = {"wav", "m4a", "mp3"}
app = Flask(__name__,
            static_folder="server/dist/public",  # 빌드된 React
            static_url_path="")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/transcribe", methods=["POST"])
def transcribe():
    # 파일이 있는지 확인
    if "file" not in request.files:
        return make_response("No file part in the request", 400)

    file = request.files["file"]

    # 파일 이름이 비어있는 경우
    if file.filename == "":
        return make_response("No selected file", 400)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # 실제 STT 모델을 돌린다고 가정 (여기서는 mock)
        mock_transcription = f"""[Mock Transcription]

        File name: "{filename}"
        
        This is a sample transcription for the uploaded audio file.
        In a production environment, this text would be generated
        by your speech-to-text (STT) model or external transcription service.
        """

        # text/plain으로 반환
        response = make_response(mock_transcription, 200)
        response.headers["Content-Type"] = "text/plain; charset=utf-8"

        # CORS 허용
        response.headers["Access-Control-Allow-Origin"] = "*"

        return response

    # 허용되지 않은 확장자
    return make_response("File type not allowed. Please upload wav/m4a/mp3.", 400)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
