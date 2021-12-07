
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

    cap = cv2.VideoCapture(Config.SOURCE)
    CAMERA_WIDTH  = cap.get(3)  # float `width`
    CAMERA_HEIGHT = cap.get(4)  # float `height`
    # size = (int(CAMERA_WIDTH), int(CAMERA_HEIGHT))
    # out = cv2.VideoWriter(str(Config.OUTPUT),cv2.VideoWriter_fourcc(*'MJPG'),10, size)
    prev_frame_time = 0
    new_frame_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        # img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640,640))
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (416,416))
        # img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        # cv2.imshow('Pi Feed', img)
        # res = detect_objects(interpreter, img, 0.2)
        boxes, scores, classes, valid_detections = yolo_detect(interpreter, img, 0.5)
        # print(boxes, scores, classes, valid_detections)

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        frame = draw_bbox(frame, pred_bbox)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(frame, str(fps), (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
        
        cv2.imshow('Pi Feed', frame)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            # out.release()
            cap.release()
            cv2.destroyAllWindows()
