import cv2, time
from app.util.yolo_tiny_detect import YoloDetection as YOLO
from app.tracker import *
from app.util.detection_roi import draw_roi, get_roi_frame
from config import Config
from app.utility import *
from app.util.bounding_box import *
from joblib import Parallel, delayed
import multiprocessing
from app.counter import attempt_count


NUM_CORES = int(multiprocessing.cpu_count() / 2)
class ObjectCounter():
    
    def __init__(self, frame, tracker):
        self.frame = frame
        self.label = read_class_names(Config.LABEL)
        self.f_height, self.f_width, _ = self.frame.shape
        self.droi = Config.DROI
        self.blobs = {}
        self.mcdf = Config.MCDF # maximum consecutive detection failures
        self.mctf = Config.MCTF # maximum consecutive tracking failures
        self.detection_interval = 1
        self.tracker = tracker
        droi_frame = get_roi_frame(self.frame)
        _bounding_boxes, _confidences, _classes,  _num_boxes = YOLO().getBoundingBox(droi_frame)
        self.pred_bbox = [_bounding_boxes, _confidences, _classes, _num_boxes]
        self.blobs = add_new_blobs(
            _bounding_boxes, _classes, _confidences,
             self.blobs, self.frame, self.tracker, self.mcdf)
        self.frame_count = 0
        self.counting_lines = Config.LINE
        self.counts = {counting_line['label']: {} for counting_line in json.loads(Config.LINE)} # counts of objects by type for each counting line
        self.show_counts = Config.DRAW_LINE

    def count(self, frame):
        self.frame = frame
        blobs_list = list(self.blobs.items())
        blobs_list = Parallel(n_jobs=NUM_CORES, prefer='threads')(
            delayed(update_blob_tracker)(blob, blob_id, self.frame) for blob_id, blob in blobs_list
        )
        self.blobs = dict(blobs_list)

        for blob_id, blob in blobs_list:
            # count object if it has crossed a counting line
            blob, self.counts = attempt_count(blob, blob_id, self.counting_lines, self.counts)
            self.blobs[blob_id] = blob

            # remove blob if it has reached the limit for tracking failures
            if blob.num_consecutive_tracking_failures >= self.mctf:
                del self.blobs[blob_id]
        
        if self.frame_count >= self.detection_interval:
            # rerun detection
            _bounding_boxes, _confidences, _classes,  _num_boxes = YOLO().getBoundingBox(self.frame)
            self.pred_bbox = [_bounding_boxes, _confidences, _classes, _num_boxes]
            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, tracker='kcf', mcdf=2)
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1
    
    def visualize(self):
        frame = self.frame
        # draw and label blob bounding boxes
        _bounding_boxes, _confidences, _classes, _num_boxes = self.pred_bbox
        index = 0
        for _id, blob in self.blobs.items():
            frame = draw_bbox(_bounding_boxes[index], frame, _confidences[index], self.label, _classes[index], _id, blob)

        if self.show_counts:
            draw_lines(frame)
            offset = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            for line, objects in self.counts.items():
                cv2.putText(frame, line, (10, 40 * offset), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                for label, count in objects.items():
                    offset += 1
                    cv2.putText(frame, "{}: {}".format(label, count), (10, 40 * offset), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                offset += 2

        return frame