
import numpy as np
import tensorflow as tf

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index=0):
  """Returns the output tensor at the given index."""
  # output_details = interpreter.get_output_details()[index]
  # tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  
  output_details = interpreter.get_output_details()
  pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
  return pred
  # return tensor

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


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details

  # pred = get_output_tensor(interpreter)
  # boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([416, 416]))

  # print("---------------------")
  # print("boxes : ",boxes) 
  # print("scores : ",pred_conf) 
  # print("classes : ",classes) 

  # Old version output detail
  # boxes = get_output_tensor(interpreter, 0)
  # classes = get_output_tensor(interpreter, 1)
  # scores = get_output_tensor(interpreter, 2)
  # count = get_output_tensor(interpreter, 3)

  # newer version output detail
  # boxes = get_output_tensor(interpreter, 1)
  # classes = get_output_tensor(interpreter, 3)
  # scores = get_output_tensor(interpreter, 0)
  # count = get_output_tensor(interpreter, 2)
  
  # print("---------------------")
  # print("boxes : ",boxes) 
  # print("classes : ",classes) 
  # print("scores : ",scores) 
  # print("count : ",count) 

  results = []
  # for i in range(len(classes)):
  #   if scores[i] >= threshold:
  #     result = {
  #         'bounding_box': list(boxes[i]),
  #         'class_id': classes[i],
  #         'score': scores[i]
  #     }
  #     results.append(result)
  
  # print(results)
  return results
