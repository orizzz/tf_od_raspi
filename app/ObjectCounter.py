import cv2, time
from app.util.yolo_tiny_detect import YoloDetection as YOLO
from app.tracker import *
from app.detection_roi import draw_roi, get_roi_frame
from config import Config
from app.utility import *
from app.util.bounding_box import *

class ObjectCounter():
    
    def __init__(self, frame):
        self.frame = frame
        self.label = read_class_names(Config.LABEL)
        self.f_height, self.f_width, _ = self.frame.shape
        self.droi = Config.DROI
        self.blobs = {}
        droi_frame = get_roi_frame(self.frame)
        _bounding_boxes, _confidences, _classes,  _num_boxes = YOLO().getBoundingBox(droi_frame)
        self.pred_bbox = [_bounding_boxes, _confidences, _classes, _num_boxes]
        self.blobs = add_new_blobs(
            _bounding_boxes, _classes, _confidences,
             self.blobs, self.frame, tracker='kcf', mcdf=2)

    def count(self, frame):
        # pass
        self.frame = frame
        # blobs_list = list(self.blobs.items())
        # print(blobs_list)
        _bounding_boxes, _confidences, _classes,  _num_boxes = YOLO().getBoundingBox(self.frame)
        self.pred_bbox = [_bounding_boxes, _confidences, _classes, _num_boxes]
        # frame = draw_bbox(coor, frame, score, self.clasess, class_ind)
        
        # return frame
    
    def visualize(self, frame):
        
        _bounding_boxes, _confidences, _classes, _num_boxes = self.pred_bbox
        for index in range(_num_boxes):
            frame = draw_bbox(_bounding_boxes[index], frame, _confidences[index], self.label, _classes[index])

        return frame