import numpy as np

def extract_xywh_SSD(faces, face_nth, frame_h, frame_w):
    pass

def extract_xywh_hog(face,):
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y
    return (x, y, w, h)