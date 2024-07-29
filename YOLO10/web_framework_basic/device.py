import json
from datetime import datetime
import cv2
import numpy as np
import requests

cap = cv2.VideoCapture(0)
t2 = datetime.now()
while True:
    t1 = t2
    t2 = datetime.now()
    time_diff = (t2 - t1).total_seconds()
    print(f"Frame time: {time_diff * 1000:.2f} ms")

    ret, frame = cap.read()
    if not ret:
        print('error')
        break

    _, buffer = cv2.imencode('.jpg', frame)
    try:
        response = requests.post("http://192.168.137.1:1000/predict", files={"image": buffer.tobytes()}, timeout=0.2)

        if response.status_code == 200:
            detections = json.loads(response.text)
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(frame, f"Class: {int(cls)}, Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    except Exception as e:
        print(e)

    cv2.rectangle(frame, (0, 0), (210, 40), (255, 255, 255), -1)
    cv2.putText(frame, f'Time: {time_diff * 1000:.2f} ms', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Annotated Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()