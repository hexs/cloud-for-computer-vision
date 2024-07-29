import json
import urllib.request
from datetime import datetime
import cv2
import numpy as np
import requests


def video_capture():

    url = f"http://192.168.225.137:1000/get-detections"
    while True:
        t1 = datetime.now()

        req = urllib.request.urlopen('http://192.168.225.137:2000/image?source=video_capture&id=0')
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        frame = cv2.imdecode(arr, -1)

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




if __name__ == "__main__":
    video_capture()