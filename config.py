from environs import Env
import base64

env = Env()
env.read_env()

class Config(object):
    INTERPRETER = env.str("INTERPRETER", default="detect.tflite")
    LABEL = env.str("LABEL", default="data/label.txt")

    CAMERA_WIDTH = env.str("CAMERA_WIDTH", default=640)
    CAMERA_HEIGHT = env.str("CAMERA_HEIGHT", default=480)