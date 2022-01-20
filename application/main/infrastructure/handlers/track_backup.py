# limit the number of cpus used by high performance libraries

import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
# sys.path.insert(0, './yolov5')
lib_path = os.path.abspath(os.path.join('yolov5'))
sys.path.append(lib_path)

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse


from util.face_visualizer import plot_face, plot_id
from util.common import  read_yml, write_csv, extract_frame_info, str_array
from util.frontal_face import hog_model, SSD_model
from util.opt_class import OPT
from util.extract_xywh import extract_xywh_hog


import platform
import shutil
from pathlib import Path
import cv2
# import dlib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta, datetime
import torch
import torch.backends.cudnn as cudnn


class Tracker:
    def __init__(self, config_path:str) -> None:

        config = read_yml(config_path)
        output = config['output']
        source = config['source']
        yolo_weights = config['yolo_weights']
        deep_sort_weights = config['deep_sort_weights']
        show_vid = config['show_vid']
        save_vid = config['save_vid']
        save_txt = config['save_txt']
        save_csv = config['save_csv']
        imgsz = config['imgsz']
        evaluate = config['evaluate']
        half = config['half']
        config_deepsort = config['config_deepsort']
        visualize = config['visualize']
        fourcc = config['fourcc']
        device = config['device']
        augment = config['augment']
        dnn = config['dnn']
        conf_thres = config['conf_thres']
        iou_thres = config['iou_thres']
        classes = config['classes']
        agnostic_nms = config['agnostic_nms']
        max_det = config['max_det']

        self.opt = OPT(output, source, yolo_weights, \
                        deep_sort_weights, show_vid, save_vid, \
                        save_txt, save_csv, imgsz, evaluate, half, \
                        config_deepsort, visualize, fourcc, \
                        device, augment, dnn, \
                        conf_thres, iou_thres, classes, \
                    agnostic_nms, max_det)

        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand

        
    def detect(self):
        opt = self.opt
        out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, save_csv, imgsz, evaluate, half = \
            opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
                opt.save_txt, opt.save_csv, opt.imgsz, opt.evaluate, opt.half

        webcam = source == '0' or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device(opt.device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
        # its own .txt file. Hence, in that case, the output folder is not restored
        if not evaluate:
            if os.path.exists(out):
                pass
                # shutil.rmtree(out)  # delete output folder
            else:
                os.makedirs(out)  # make new output folder

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(opt.yolo_weights, device=device, dnn=opt.dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        frontal_face = "hog"
        if frontal_face == "SSD":
            face_model = SSD_model(680,480)
            
        elif frontal_face == "hog":
            face_model = hog_model()

        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()

        # Set Dataloader
        vid_path, vid_writer = None, None
        # Check if environment supports image displays
        if show_vid:
            show_vid = check_imshow()

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        save_path = str(Path(out))
        # extract what is in between the last '/' and last '.'
        txt_file_name = source.split('/')[-1].split('.')[0]
        txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
        csv_path = str(Path(out)) + '/' + txt_file_name + '.csv'
        df_tl = pd.DataFrame(columns=['time', 'frame', 'n_pp_at_time', 'n_attentive_pp_at_time',\
             'IDs_people', 'IDs_attention', 'bb_people', 'bb_attention',\
                'total_pp_appear_uptotime', 'total_attentive_pp_appear_uptotime', ])
        if save_csv:
            with open(csv_path, 'w') as f:
                f.write("time;n_pp_at_time;looking_face_count;IDs_people;IDs_looking;bb_people;bb_looking \n")

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        # list_ouputs = {}
        # list_frontal_faces = {}
        list_duration = {} # id:{start in view, exit view, pass duration, attention duration }
        list_pp = set()
        list_face = set()

        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            frame_height = im0s.shape[0]
            frame_width = im0s.shape[1]
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
            pred = model(img, augment=opt.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    # s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # s += '%gx%g ' % img.shape[2:]  # print string
                save_path = str(Path(out) / Path(p).name)

                annotator = Annotator(im0, line_width=2, pil=not ascii)

                # Predict face
                # faces, len_faces = face_model.process(im0)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    last_row = len(df_tl)
                    df_tl.loc[last_row,'frame'] = frame_idx
                    df_tl.loc[last_row,'n_pp_at_time'] = len(outputs)
                    if len(outputs)>0:
                        # print(outputs)
                        df_tl['IDs_people'][last_row] = list(outputs[:, 4])
                        df_tl['bb_people'][last_row] = list(outputs[:, :4])
                    
                    if last_row >= 2:
                        previous_IDs = df_tl['IDs_people'][last_row-1]
                        current_IDs = df_tl['IDs_people'][last_row]
                        
                        if np.isnan(np.sum(previous_IDs)):
                            previous_IDs = []
                        if np.isnan(np.sum(current_IDs)):
                            current_IDs = []
                            
                        for ID in current_IDs:
                            if (ID not in previous_IDs) and (ID not in list_pp):
                                list_duration[ID] = {}
                                list_duration[ID]['in'] = frame_idx
                                list_duration[ID]['exit'] = -1
                                list_duration[ID]['start_attend'] = -1
                                list_duration[ID]['end_attend'] = -1
                                
                        for ID in previous_IDs:
                            if (ID not in current_IDs):
                                list_duration[ID]['exit'] = frame_idx
                    

                    # save output info
                    # list_ouputs[frame_idx] = outputs
                    
                    
                    face_outputs = []  # for faces
                    if len(outputs) > 0:

                        for j, (output, conf) in enumerate(zip(outputs, confs)): 
                            
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)  # integer class
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                           

                            # get upper part of person's box for detect face and visualize box
                            bbox_left, bbox_top, bbox_right, bbox_bottom = [output[i] for i in range(4)]

                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]

                            upper_body = im0[bbox_top:(bbox_top+bbox_h//2), bbox_left:bbox_right]
                            faces, len_faces = face_model.process(upper_body)
                            
                            if len_faces != 0:
                                for face in faces:
                                    x,y,w,h = extract_xywh_hog(face)
                                    x += bbox_left
                                    y += bbox_top                            
                                    plot_face(x,y,w,h, im0, frame_height, frame_width)
                                    plot_id(x,y,w,h, id, im0, frame_height, frame_width)

                                    face_outputs.append([x,y,w,h,id])
                                    if list_duration[id]['start_attend'] == -1:
                                        list_duration[id]['start_attend'] = frame_idx
                                    else:
                                        list_duration[id]['end_attend'] = frame_idx

                        face_outputs = np.asarray(face_outputs)           
                        

                        df_tl.loc[last_row,'n_attentive_pp_at_time'] = len(face_outputs)
                        if len(face_outputs)>0:
                            df_tl['IDs_attention'][last_row] = list(face_outputs[:, 4])
                            df_tl['bb_attention'][last_row] = list(face_outputs[:, :4])


                        # list_frontal_faces[frame_idx] = np.asarray(face_outputs)
            

                        pp_count, face_count, IDs_pp, IDs_face =\
                            df_tl.iloc[-1]['n_pp_at_time'], df_tl.iloc[-1]['n_attentive_pp_at_time'],\
                                df_tl.iloc[-1]['IDs_people'], df_tl.iloc[-1]['IDs_attention']
                            # extract_frame_info(frame_idx, list_ouputs, list_frontal_faces)
                        LOGGER.info("{}: {} people, {} is looking".format(s, pp_count, face_count))

                        if not np.isnan(np.sum(IDs_pp)):
                            list_pp.update(list(IDs_pp)) 
                        if not np.isnan(np.sum(IDs_face)):
                            list_face.update(list(IDs_face))
                            
                    df_tl.loc[last_row, 'total_pp_appear_uptotime'] = len(list_pp)
                    df_tl.loc[last_row, 'total_attentive_pp_appear_uptotime'] = len(list_face)

           

                else:
                    deepsort.increment_ages()

                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            

                # Stream results
                im0 = annotator.result()
                if show_vid:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                else:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)

        print(list_duration)
        
        df_tl['time'] = (df_tl['frame']/fps).apply(lambda x: str(timedelta(seconds=x)))

        df_dur = pd.DataFrame(list_duration).T.reset_index()
        df_dur['duration_attention'] = df_dur['end_attend'] - df_dur['start_attend']
        df_dur.drop(columns=['end_attend', 'start_attend'], inplace=True)
        df_dur.columns = ['ID', 'in_time', 'exit_time', 'duration_attention']
        end_time =  str(timedelta(seconds=frame_idx/fps))
        for col in df_dur.columns[1:]:
            df_dur[col] = (df_dur[col]/fps).apply(lambda x: str(timedelta(seconds=x)) if x>0 else end_time)
                

        
        # write to csv file
        if save_csv:
            df_dur.to_csv(out+'/'+'duration.csv')
            df_tl.to_csv(out+'/'+'timeline.csv')
            
    
        LOGGER.info("Total of {} people passed and {} people looked at the banner".format(len(list_pp), len(list_face)))

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_vid or save_csv:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)

        engine = create_engine('postgresql+pg8000://postgres:root2351@localhost:5432/face_store')
        # df_dur = pd.read_csv(out+'/'+'duration.csv').iloc[:, 1:]
        # df_tl = pd.read_csv(out+'/'+'timeline.csv').iloc[:, 1:] 
        df_dur.to_sql("person", engine, if_exists='replace')
        df_tl.to_sql("front_store", engine, if_exists='replace')
        
        

if __name__ == '__main__':

    tracker = Tracker(config_path='../settings/config.yml')
    
    with torch.no_grad():
        tracker.detect()

