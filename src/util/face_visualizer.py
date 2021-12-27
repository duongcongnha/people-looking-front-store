import cv2
import numpy as np

custom_green = (2, 115, 36)
font = cv2.FONT_HERSHEY_SIMPLEX

def plot_face(x,y,w,h, img, frame_h, frame_w):
    cv2.rectangle(img, (x,y), (x+w,y+h), custom_green, 2)
    cv2.rectangle(img, (x, y+h), (int(x+1.5*w),int(y+h+frame_h/35)), custom_green, -1)
    cv2.putText(img, "is looking", (x, int(y+h+frame_h/35)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def plot_id(x,y,w,h, id, img, frame_h, frame_w):
    cv2.rectangle(img, (x, int(y-frame_h/35)), (int(x+w*0.82),y), custom_green, -1)
    cv2.putText(img, "id "+str(id), (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        


def dnn_face_visualize(faces, outputs, img, temp_info, frame_height, frame_width): 
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.55:
            box = faces[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (x, y, x1, y1) = box.astype("int")
            w = x1-x
            h = y1 - y
            plot_face(x,y,w,h, img, frame_height, frame_width)        


            if len(outputs) > 0:
                for output in outputs:
                    id = output[4]

                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]

                    if (x > bbox_left) and (y > bbox_top) and (x+w < bbox_left+bbox_w) and (y <(bbox_top+bbox_h)-5/6*bbox_h): #(y+h <(bbox_top+bbox_h)-1/2*bbox_h)
                        
                        temp_info.append([x,y,w,h,id])
                        plot_id(x,y,w,h, id, img, frame_height, frame_width)
    return temp_info


def hog_face_visualize(faces, outputs, img, temp_info, frame_height, frame_width): 
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        plot_face(x,y,w,h, img, frame_height, frame_width)
        
        if len(outputs) > 0:
            for output in outputs:
                id = output[4]

                bbox_left = output[0]
                bbox_top = output[1]
                bbox_w = output[2] - output[0]
                bbox_h = output[3] - output[1]
        
                if (x > bbox_left) and (y > bbox_top) and (x+w < bbox_left+bbox_w) and (y+h <(bbox_top+bbox_h)-1/2*bbox_h):
                    # list_frontal_faces.append((frame_idx, [x,y,w,h,id]))
                    temp_info.append([x,y,w,h,id])
                    plot_id(x,y,w,h, id, img, frame_height, frame_width)
    return temp_info

def face_visualize(faces, outputs, img, temp_info, frame_height, frame_width, frontal_face): 
    if frontal_face == "SSD":
        return dnn_face_visualize(faces, outputs, img, temp_info, frame_height, frame_width)
        
    elif frontal_face == "hog":
        return hog_face_visualize(faces, outputs, img, temp_info, frame_height, frame_width)

