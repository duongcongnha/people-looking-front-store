import yaml
import numpy as np
from typing import Union

def str_array (array: np.ndarray):
  output_string = ""
  for i in range(len(array)):
    output_string += str(list(array[i]))+","
  return output_string[:-1]


def read_yml(path:str):
  with open(path, 'r') as file:
      text = yaml.safe_load(file)
  return text

def update_config(config_path:str, new_config:dict):
   config_text = read_yml(config_path)
   for setting in new_config.keys():
      config_text[setting] = new_config[setting]
   with open(config_path, "w") as f:
         yaml.dump(config_text, f)

def extract_xywh(faces, face_nth, frame_height, frame_width):
  box = faces[0, 0, face_nth, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
  (x, y, x1, y1) = box.astype("int")
  w = x1-x
  h = y1 - y
  return (x, y, w, h)

def write_csv(csv_path:str, list_ouputs:dict, list_frontal_faces:dict):
  fi_list = list(set(list(list_ouputs.keys()) + list(list_frontal_faces.keys())))
  fi_list.sort()
  with open(csv_path, 'a') as f:
      for fi in fi_list:
          pp_component = list_ouputs[fi]
          face_component = list_frontal_faces[fi]

          pp_count = len(pp_component)
          face_count = len(face_component)

          if len(pp_component)>0:
              IDs_pp = list(pp_component[:,4])
              bb_pp = pp_component[:,:4]
          else:
              IDs_pp = ""
              bb_pp = ""

          if len(face_component)>0:
              IDs_face = list(face_component[:,4])      
              bb_face = face_component[:,:4]
          else:
              IDs_face = ""
              bb_face = ""

          f.write(str(fi)+";")
          f.write(str(pp_count)+";")
          f.write(str(face_count)+";")
          f.write(str(IDs_pp)+";")
          f.write(str(IDs_face)+";")
          f.write(str_array(bb_pp)+";")
          f.write(str_array(bb_face)+"\n")

