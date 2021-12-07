
import cv2, time
from tflite_runtime.interpreter import Interpreter
from app.object_detection import detect_objects
from app.yolo_tiny_detect import detect_objects as yolo_detect
from config import Config
from app.utility import *

CAMERA_WIDTH = int(Config.CAMERA_WIDTH)
CAMERA_HEIGHT = int(Config.CAMERA_HEIGHT)

def detect():
    labels = load_labels(Config.LABEL)
    interpreter = Interpreter(Config.INTERPRETER)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    source = Config.SOURCE
    if Config.SOURCE == "0" or Config.SOURCE == "NONE":
        source = 0

    cap = cv2.VideoCapture(source)

    if Config.OUTPUT:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(Config.OUTPUT, codec, 25, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (416,416))
        boxes, scores, classes, valid_detections = yolo_detect(interpreter, img, 0.5)

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        frame = draw_bbox(frame, pred_bbox)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(frame, str(fps), (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
        
        # cv2.imshow('Pi Feed', frame)
        out.write(frame)

        if 0xFF ==ord('q'):
            out.release()
            cap.release()
            cv2.destroyAllWindows()
