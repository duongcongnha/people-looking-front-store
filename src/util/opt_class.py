class OPT:
    def __init__(self, output, source, yolo_weights, \
                    deep_sort_weights, show_vid, save_vid, \
                    save_txt, save_csv, imgsz, evaluate, half, \
                    config_deepsort, visualize, fourcc, \
                    device, augment, dnn, \
                    conf_thres, iou_thres, classes, \
                    agnostic_nms, max_det) -> None:
        
        
        self.output, self.source, self.yolo_weights, \
        self.deep_sort_weights, self.show_vid, self.save_vid, \
        self.save_txt, self.save_csv, self.imgsz, self.evaluate, self.half, \
        self.config_deepsort, self.visualize, self.fourcc, \
        self.device, self.augment, self.dnn, \
        self.conf_thres, self.iou_thres, self.classes, \
        self.agnostic_nms, self.max_det = \
        \
        output, source, yolo_weights, \
        deep_sort_weights, show_vid, save_vid, \
        save_txt, save_csv, imgsz, evaluate, half, \
        config_deepsort, visualize, fourcc, \
        device, augment, dnn, \
        conf_thres, iou_thres, classes, \
        agnostic_nms, max_det



