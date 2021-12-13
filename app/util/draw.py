import cv2, colorsys, json
from config import Config
import numpy as np


class DrawUtils():
    def __init__(self, frame):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.frame = frame
        self.image_h, self.image_w, _ = frame.shape
        self.line_thick = int(0.6 * (self.image_h + self.image_w) / 600)


    def draw_bbox(self, _id, blob, show_label=True):
        frame = self.frame
        num_classes = len(Config.LABEL)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        fontScale = 0.3
        line_type = cv2.LINE_AA
        label = blob.type
        bbox_color = colors[blob.type_value]

        coor = [int(v) for v in blob.bounding_box]
        c1 = (int(coor[0]), int(coor[1]))
        c2 = (int(coor[2]), int(coor[3]))
        # c3 = (int(coor[2]), int(coor[1]))
        # c4 = (int(coor[0]), int(coor[3]))

        """
        bounding box coor example

        c1             c3
        |+++++++++++++++|
        | bounding box  |
        |               |
        |+++++++++++++++|
        c4             c2

        """

        cv2.rectangle(frame, c1, c2, bbox_color, self.line_thick)

        if show_label:
            bbox_mess = '{clasess}: {score:.0f}%'.format(clasess=label, score=blob.type_confidence*100)

            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=self.line_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(frame, c1, (int(np.float32(c3[0])), int(np.float32(c3[1]))), bbox_color, -1) #filled

            cv2.putText(frame, bbox_mess, (int(c1[0]), int(np.float32(c1[1] - 2))), self.font,
                        fontScale, (255, 255, 255), self.line_thick // 2, line_type)

        object_label = f'ID: {_id[:8]}'
        cv2.putText(frame, object_label, (int(c1[0]), int(np.float32(c2[1] + 5))), self.font,
            fontScale, (255, 255, 255), self.line_thick // 2, line_type)
    
    def draw_line(self):
        frame = self.frame
        LINE = json.loads(Config.LINE)
        for item in LINE:
            cv2.line(frame, item['line'][0], item['line'][1], (0,255,0), self.line_thick)
            cl_label_origin = (item['line'][0][0], item['line'][0][1] + 35)
            cv2.putText(frame, item['label'], cl_label_origin, self.font, 1, (0,255,0), self.line_thick, cv2.LINE_AA)

    def draw_counters(self, counts):
        frame = self.frame
        fontScale = 1
        offset = 1
        for line, objects in counts.items():
            cv2.putText(frame, line, (10, 40 * offset), self.font, fontScale, (0, 0, 0), self.line_thick, cv2.LINE_AA)
            for label, count in objects.items():
                offset += 1
                cv2.putText(frame, "{}: {}".format(label, count), (10, 40 * offset), self.font, fontScale, (0, 0, 0), self.line_thick, cv2.LINE_AA)
            offset += 2