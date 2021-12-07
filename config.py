from environs import Env
import base64

env = Env()
env.read_env()

class Config(object):
    INTERPRETER = env.str("INTERPRETER", default="detect.tflite")
    LABEL = env.str("LABEL", default="data/label.txt")

    CAMERA_WIDTH = env.str("CAMERA_WIDTH", default=640)
    CAMERA_HEIGHT = env.str("CAMERA_HEIGHT", default=480)
    INPUT_SIZE = env.str("INPUT_SIZE", default=416)
    IOU = env.str("IOU", default=0.5)
    THERSHOLD = env.str("THERSHOLD", default=0.5)

    SOURCE = env.str("SOURCE")
    OUTPUT = env.str("OUTPUT")
    LINE = env.str("LINE")
    DETECTOR = env.str("DETECTOR", default="TF")