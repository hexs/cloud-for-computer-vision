import asyncio
import websockets
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("../yolov10n.pt")


async def process_frame(websocket, path):
    async for message in websocket:
        # Decode the image
        nparr = np.frombuffer(message, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(frame)

        # Instead of sending back the full image, send only the detection results
        detections = results[0].boxes.data.tolist()
        await websocket.send(str(detections))


start_server = websockets.serve(process_frame, "192.168.137.1", 1000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
