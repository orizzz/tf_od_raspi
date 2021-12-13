import json, cv2, random, colorsys
from config import Config
import numpy as np

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for index, name in enumerate(data):
            names[index] = name.strip('\n')
    return names

def draw_lines(frame, lines=Config.LINE):
    LINE = json.loads(lines)
    for item in LINE:
      start = (item["line"][0][0], item["line"][0][1])
      end = (item["line"][1][0], item["line"][1][1])
      cv2.line(frame, start, end, (0,255,0), 3)

def draw_bbox(coor, image, score, classes, class_ind, _id, blob, show_label=True):
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    label = classes[class_ind["value"]]
    image_h, image_w, _ = image.shape

    fontScale = 0.3
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    bbox_color = colors[class_ind["value"]]
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    coor = [int(v) for v in blob.bounding_box]
    c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
    cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

    if show_label:
        bbox_mess = '{clasess}: {score:.0f}%'.format(clasess=label, score=score*100)

        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(image, c1, (int(np.float32(c3[0])), int(np.float32(c3[1]))), bbox_color, -1) #filled

        cv2.putText(image, bbox_mess, (int(c1[0]), int(np.float32(c1[1] - 2))), font,
                    fontScale, (255, 255, 255), bbox_thick // 2, line_type)

        object_label = f'Id: {_id[:8]}'
        cv2.putText(image, object_label, (int(c1[0]), int(np.float32(c2[1] + 5))), font,
            fontScale, (255, 255, 255), bbox_thick // 2, line_type)

    return image