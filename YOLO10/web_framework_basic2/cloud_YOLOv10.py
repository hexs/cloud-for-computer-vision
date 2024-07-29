# ประมวลผลได้มากกว่า ทำให้ fps สูงกว่า
# แต่ข้อมูลที่ตอบกลับเป็นข้อมูลเก่าที่ประมวลผลเก็บไว้

import json
import urllib.request
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2
import numpy as np
from multiprocessing import Process, Manager

# Load the YOLO model
model = YOLO("../yolov10n.pt")

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict")
def predict():
    data = app.config['data']
    detections = data['detections']
    return json.dumps(detections)


def get_image(data):
    url = f"http://{data['client_ipv4'][0]}:{data['client_ipv4'][1]}/image"
    while True:
        t1 = datetime.now()
        try:
            req = urllib.request.urlopen(url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)

            results = model(img)
            detections = results[0].boxes.data.tolist()
        except Exception as e:
            print(f"Error: {e}")
            detections = []
        data['detections'] = detections
        t2 = datetime.now()
        time_diff = (t2 - t1).total_seconds()
        print(f"Processing time: {time_diff:.2f} seconds")


def run_server(data):
    app.config['data'] = data
    app.run(*data['server_ipv4'], debug=True, use_reloader=False)


if __name__ == "__main__":
    manager = Manager()
    data = manager.dict()
    data['detections'] = []
    data['client_ipv4'] = '192.168.137.1', 2000
    data['server_ipv4'] = '192.168.137.1', 1000

    get_image_process = Process(target=get_image, args=(data,))
    run_server_process = Process(target=run_server, args=(data,))

    get_image_process.start()
    run_server_process.start()

    get_image_process.join()
    run_server_process.join()
