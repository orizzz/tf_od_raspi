import json, cv2, random, colorsys
from config import Config
import numpy as np

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for index, name in enumerate(data):
            names[index] = name.strip('\n')
    return names

def draw_lines(frame):
  LINE = json.loads(Config.LINE)
  for item in LINE:
    start = (item["line"][0][0], item["line"][0][1])
    end = (item["line"][1][0], item["line"][1][1])
    cv2.line(frame, start, end, (0,255,0), 3)

def draw_bbox(image, bboxes, classes=read_class_names(Config.LABEL), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    # colors = (255,255,0)

    # random.seed(0)
    # random.shuffle(colors)
    # random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes

    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.3
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '{clasess}: {score:.0f}%'.format(
              clasess=classes[class_ind], 
              score=score*100)

            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (int(np.float32(c3[0])), int(np.float32(c3[1]))), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (int(c1[0]), int(np.float32(c1[1] - 2))), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image