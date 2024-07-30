# ประมวลผลได้มากกว่า ทำให้ fps สูงกว่า
# แต่ข้อมูลที่ตอบกลับเป็นข้อมูลเก่าที่ประมวลผลเก็บไว้

import json
import os
import urllib.request
from datetime import datetime
import socket
from urllib.parse import urlparse
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from multiprocessing import Process, Manager


def update_config_file(filename, updates):
    data = {}
    if filename in os.listdir():
        with open(filename, 'r') as file:
            data = json.load(file)
        data.update(updates)

    if data.get('ipv4') is None:
        hostname = socket.gethostname()
        ipv4 = socket.gethostbyname(hostname)
        data.update({'ipv4': ipv4})
    if data.get('port') is None:
        data.update({'port': '1000'})

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(data)
    return data


# Load the YOLO model
model = YOLO("../yolov10n.pt")

app = Flask(__name__)


@app.route("/")
def hello_world():
    return (
        "<a href='/config'>config</a><br>"
        "<a href='/reset'>reset</a><br>"
        "<a href='/get-detections'>get-detections</a><br>"
    )


@app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        ipv4 = request.form.get('ipv4')
        port = request.form.get('port')
        other = request.form.get('other')

        app.config['data']['url_image'] = f"http://{ipv4}:{port}/{other}"
        print(f"url = http://{ipv4}:{port}/{other}")

        return f"http://{ipv4}:{port}/{other}"

    parsed_url = urlparse(app.config['data']['url_image'])
    ipv4 = parsed_url.hostname
    port = parsed_url.port
    return render_template('config.html', default_ipv4=ipv4, default_port=port)


@app.route("/reset")
def reset():
    app.config['data']['url_image'] = None
    return (
        "<a href='/config'>config</a><br>"
        "<a href='/reset'>reset</a><br>"
        "<a href='/get-detections'>get-detections</a><br>"
    )


@app.route("/get-detections")
def get_detections():
    detections = app.config['data']['detections']
    return json.dumps(detections)


def predict(data):
    while True:
        if data['url_image']:
            t1 = datetime.now()
            try:
                req = urllib.request.urlopen(data['url_image'])
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
    app.run(data['ipv4'], data['port'], debug=True, use_reloader=False)


if __name__ == "__main__":
    config = update_config_file('config.json', {})

    manager = Manager()
    data = manager.dict()
    data['detections'] = []
    data['ipv4'] = f"{config['ipv4']}"
    data['port'] = f"{config['port']}"
    data['url_image'] = None

    get_image_process = Process(target=predict, args=(data,))
    run_server_process = Process(target=run_server, args=(data,))

    get_image_process.start()
    run_server_process.start()

    get_image_process.join()
    run_server_process.join()
