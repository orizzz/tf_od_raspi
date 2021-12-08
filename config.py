from environs import Env
import base64

env = Env()
env.read_env()

class Config(object):
    INTERPRETER = env.str("INTERPRETER", default="detect.tflite")
    LABEL = env.str("LABEL", default="data/label.txt")

    CAMERA_WIDTH = env.int("CAMERA_WIDTH", default=640)
    CAMERA_HEIGHT = env.int("CAMERA_HEIGHT", default=480)
    INPUT_SIZE = env.int("INPUT_SIZE", default=416)
    IOU = env.float("IOU", default=0.5)
    THERSHOLD = env.float("THERSHOLD", default=0.5)

    SOURCE = env.str("SOURCE", default=0)
    OUTPUT = env.str("OUTPUT", default=None)
    VISUALIZE = env.bool("VISUALIZE", default=True)
    LINE = env.str("LINE")
    DETECTOR = env.str("DETECTOR", default="TF")