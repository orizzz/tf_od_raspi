
import cv2
from tflite_runtime.interpreter import Interpreter
from app.object_detection import detect_objects
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
    
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        # cv2.imshow('Pi Feed', img)
        res = detect_objects(interpreter, img, 0.2)


        # draw_lines(frame)
        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            # print(ymin, xmin, ymax, xmax)
            xmin = int(max(1, xmin * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            # print(ymin, xmin, ymax, xmax)
            
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),2)
            score = "{:.2f}".format(result['score'])
            label = "{} {}".format(labels[int(result['class_id'])], score)
            cv2.putText(
                frame,
                label,
                (xmin, ymin-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2,cv2.LINE_AA)
        # out.write(frame)
        # out.write(frame)
        cv2.imshow('Pi Feed', frame)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            # out.release()
            cap.release()
            cv2.destroyAllWindows()
