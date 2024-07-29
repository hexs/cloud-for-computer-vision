import json
import socket
from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2
import numpy as np

model = YOLO("../yolov10n.pt")

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame)
    detections = results[0].boxes.data.tolist()

    return json.dumps(detections)


if __name__ == "__main__":
    hostname = socket.gethostname()
    ipv4_address = socket.gethostbyname(hostname)
    app.run('192.168.137.1', 1000, True)

'''
1 client send data to server
2 server predict (wait perdict) and send detections to client

i want multiprocessing
1 client send data1 to server
2 client send data2 to server
3 server send old detections(detections1) to client
4 client send data3 to server
5 server send old detections(detections2) to client




'''
