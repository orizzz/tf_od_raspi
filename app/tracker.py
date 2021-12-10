import cv2

class ObjectTracker():
    def __init__(self):
        pass

    def track_object(self, bounding_box, frame):
        '''`
        Create an OpenCV CSRT Tracker object.
        '''
        bbox = tuple(int(x) for x in bounding_box)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        return tracker