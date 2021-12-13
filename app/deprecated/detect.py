
import cv2, time
from app.util.yolo_tiny_detect import YoloDetection
# from app.tracker import ObjectTracker
from app.util.detection_roi import draw_roi, get_roi_frame
from config import Config
from app.utility import *

CAMERA_WIDTH = Config.CAMERA_WIDTH
CAMERA_HEIGHT = Config.CAMERA_HEIGHT

def detect():
    labels = read_class_names(Config.LABEL)

    YOLO = YoloDetection()
    
    source = Config.SOURCE
    if Config.SOURCE == "0" or Config.SOURCE == "NONE" or not Config.SOURCE:
        source = 0

    cap = cv2.VideoCapture(source)

    if Config.OUTPUT:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(Config.OUTPUT, codec, 25, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if Config.USE_DROI:
            droi_frame = get_roi_frame(frame)
            img = cv2.resize(cv2.cvtColor(droi_frame, cv2.COLOR_BGR2RGB), (416,416))
        else:
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (416,416))
        

        boxes, scores, classes, valid_detections = YOLO.getBoundingBox(img)

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        frame = process(frame, pred_bbox, classes=labels)

        if Config.USE_DROI and Config.DRAW_ROI:
            frame = draw_roi(frame, Config.DROI)
        
        if Config.DRAW_LINE:
            draw_lines(frame)

        if Config.OUTPUT:
            out.write(frame)

        if Config.VISUALIZE:
            cv2.imshow('Pi Feed', frame)
            if cv2.waitKey(10) & 0xFF ==ord('q'):
                out.release()
                cap.release()
                cv2.destroyAllWindows()

def process(frame, bboxes, classes):
    out_boxes, out_scores, out_classes, num_boxes = bboxes

    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > len(classes): continue
        coor = out_boxes[0][i]
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])

        frame_h, frame_w, _ = frame.shape
        coor[0] = int(coor[0] * frame_h)
        coor[2] = int(coor[2] * frame_h)
        coor[1] = int(coor[1] * frame_w)
        coor[3] = int(coor[3] * frame_w)
        # print(coor)

        # tracker = ObjectTracker().track_object(coor, frame)
        # (success, box) = tracker.update(frame)
        # box = updatedBbox(box)
        # print(success, box)
        # print(coor)

        # frame = draw_bbox(box, frame, score, classes, class_ind)
        frame = draw_bbox(coor, frame, score, classes, class_ind)
    
    return frame

    
