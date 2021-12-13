import uuid
import random


def generate_object_id():
    return str(uuid.uuid4())[0:8]
