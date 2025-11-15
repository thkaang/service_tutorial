# app.py
from flask import Flask, send_from_directory, jsonify, request
import os

app = Flask(__name__,
            static_folder="dist/public",  # 빌드된 React
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
