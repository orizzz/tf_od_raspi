import numpy as np
import tensorflow as tf
import cv2
from config import Config

input_size = int(Config.INPUT_SIZE)


def detect_objects(interpreter, image, threshold):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    image_data = image / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    # if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
    #     boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
    #                                     input_shape=tf.constant([input_size, input_size]))
    # else:
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                    input_shape=tf.constant([input_size, input_size]))
    # print(boxes, pred_conf)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold= float(Config.IOU),
        score_threshold= float(Config.THERSHOLD)
    )
    return boxes, scores, classes, valid_detections

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)