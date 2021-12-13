# from app.detect import detect
from app.detect import detect
from app.ObjectCounter import ObjectCounter
import cv2, time
from app.util.yolo_tiny_detect import YoloDetection
# from app.tracker import ObjectTracker
from app.detection_roi import draw_roi, get_roi_frame
from config import Config
from app.utility import *

def run():
    labels = read_class_names(Config.LABEL)
    source = Config.SOURCE
    output = Config.OUTPUT

    YOLO = YoloDetection()
    
    if Config.SOURCE == "0" or Config.SOURCE == "NONE" or not Config.SOURCE:
        source = 0

    cap = cv2.VideoCapture(source)
    retval, frame = cap.read()
    f_height, f_width, _ = frame.shape

    objectCounter = ObjectCounter(frame)

    if output:
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output, codec, 25, (f_width, f_height))
    
    while retval:
        
        # YOLO.getBoundingBox(frame)
        objectCounter.count(frame)
        frame = objectCounter.visualize(frame)

        if Config.VISUALIZE:
            cv2.imshow('Pi Feed', frame)
            if cv2.waitKey(10) & 0xFF ==ord('q'):
                out.release()
                cap.release()
                cv2.destroyAllWindows()
        retval, frame = cap.read()

if __name__ == "__main__":
    run()
    # detect()