import cv2
import asyncio
import websockets
import numpy as np
import json


async def send_frames():
    uri = "ws://192.168.137.1:1000"
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame to jpg
            _, buffer = cv2.imencode('.jpg', frame)

            # Send frame to server
            await websocket.send(buffer.tobytes())

            # Receive detection results
            detection_results = await websocket.recv()
            detections = json.loads(detection_results)

            # Draw detections on the frame
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Class: {int(cls)}, Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Frame with Detections', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


asyncio.get_event_loop().run_until_complete(send_frames())