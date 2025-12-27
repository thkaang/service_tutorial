import os
import requests
from flask import Flask, send_from_directory, jsonify, request, make_response

SECRET_ENDPOINT = "http://127.0.0.1:5000/transcribe"
ALLOWED_EXTENSIONS = ["wav", "m4a", "mp3"]
app = Flask(__name__,
            static_folder="server/dist/public",  # 빌드된 React
            static_url_path="")


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
    language1 = request.form["language1"]
    language2 = request.form["language2"]
    print(f"language1: {language1}, language2: {language2}")

    if file is None or file.filename == "":
        return make_response("No selected file", 400)

    files = {
        "file": (file.filename, file.stream, file.content_type)
    }

    response = requests.post(SECRET_ENDPOINT, files=files)
    response = make_response(response.content, response.status_code)
    response.headers["Content-Type"] = response.headers.get(
        "Content-Type",
        "text/plain; charset=utf-8",
    )
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
