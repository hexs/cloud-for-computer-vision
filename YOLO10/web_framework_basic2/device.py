import json
from datetime import datetime
import cv2
import numpy as np
import requests
from multiprocessing import Process, Manager
from flask import Flask, Response

app = Flask(__name__)


@app.route('/image')
def get_image():
    frame = app.config['data']['img']
    success = app.config['data']['status']
    if not success:
        frame = np.full([500, 500, 3], [200, 200, 200], np.uint8)
        cv2.putText(frame, 'Failed to capture image', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'from camera 0', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


def video_capture(data):
    cap = cv2.VideoCapture(0)
    url = f"http://{data['server_ipv4'][0]}:{data['server_ipv4'][1]}/predict"
    while True:
        t1 = datetime.now()
        status, img = cap.read()
        data['status'] = status
        if status:
            data['img'] = img.copy()
        else:
            cap = cv2.VideoCapture(0)
            data['img'] = np.full((480, 640, 3), (50, 50, 50), dtype=np.uint8)

        frame = data['img']
        response = requests.get(url)
        if response.status_code == 200:
            detections = json.loads(response.text)
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(frame, f"Class: {int(cls)}, Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        t2 = datetime.now()
        time_diff = (t2 - t1).total_seconds()
        cv2.rectangle(frame, (0, 0), (210, 40), (255, 255, 255), -1)
        cv2.putText(frame, f'Time: {time_diff * 1000:.2f} ms', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Annotated Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def run_server(data):
    app.config['data'] = data
    app.run(*data['client_ipv4'], debug=True, use_reloader=False)


if __name__ == "__main__":
    manager = Manager()
    data = manager.dict()
    data['client_ipv4'] = '192.168.137.1', 2000
    data['server_ipv4'] = '192.168.137.1', 1000
    data['img'] = None
    data['status'] = False

    capture_process = Process(target=video_capture, args=(data,))
    run_server_process = Process(target=run_server, args=(data,))

    capture_process.start()
    run_server_process.start()

    capture_process.join()
    run_server_process.join()
