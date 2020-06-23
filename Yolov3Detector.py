import math
import time
import os
import cv2
from utils import *
import copy

def compute_iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes
    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1[0]-box1[2]/2,  box1[1]-box1[3]/2, box1[0]+box1[2]/2,  box1[1]+box1[3]/2
    b2_x0, b2_y0, b2_x1, b2_y1 = box2[0]-box2[2]/2,  box2[1]-box2[3]/2, box2[0]+box2[2]/2,  box2[1]+box2[3]/2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_w = max((int_x1 - int_x0), 0)
    int_h = max((int_y1 - int_y0), 0)
    int_area = int_w * int_h

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

class YOLOV3_Detector:
    def __init__(self, dllname, configPath, weightPath, metaPath):
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.dllname = dllname
        self.configPath = configPath
        self.weightPath = weightPath
        self.metaPath = metaPath

    def loadNames(self):
        if self.altNames is None:
            # In Python 3, the metafile default access craps out on Windows (but not Linux)
            # Read the names file and create a list to feed to detect
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

    def loadConfig(self):
        configPath = self.configPath
        weightPath = self.weightPath
        metaPath = self.metaPath
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")
        if self.netMain is None:
            self.netMain = self.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = self.load_meta(metaPath.encode("ascii"))

    def loadDLL(self):
        self.hasGPU = True
        if os.name == "nt":
            cwd = os.path.dirname(__file__)
            os.environ['PATH'] = cwd + ';' + os.environ['PATH']
            winGPUdll = os.path.join(cwd, "%s.dll" % self.dllname)
            winNoGPUdll = os.path.join(cwd, "%s_nogpu.dll" % self.dllname)
            envKeys = list()
            for k, v in os.environ.items():
                envKeys.append(k)
            try:
                try:
                    tmp = os.environ["FORCE_CPU"].lower()
                    if tmp in ["1", "true", "yes", "on"]:
                        raise ValueError("ForceCPU")
                    else:
                        print("Flag value '" + tmp + "' not forcing CPU mode")
                except KeyError:
                    # We never set the flag
                    if 'CUDA_VISIBLE_DEVICES' in envKeys:
                        if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                            raise ValueError("ForceCPU")
                    try:
                        global DARKNET_FORCE_CPU
                        if DARKNET_FORCE_CPU:
                            raise ValueError("ForceCPU")
                    except NameError:
                        pass
                if not os.path.exists(winGPUdll):
                    raise ValueError("NoDLL")
                try:
                    self.lib = CDLL(winGPUdll, RTLD_GLOBAL)
                except WindowsError:
                    hasGPU = False
                    if os.path.exists(winNoGPUdll):
                        self.lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
                        print("Notice: CPU-only mode")
            except (KeyError, ValueError):
                hasGPU = False
                if os.path.exists(winNoGPUdll):
                    self.lib = CDLL(winNoGPUdll, RTLD_LOCAL)
                    print("Notice: CPU-only mode")
                else:
                    # Try the other way, in case no_gpu was
                    # compile but not renamed
                    self.lib = CDLL(winGPUdll, RTLD_LOCAL)
                    print(
                        "Environment variables indicated a CPU run, but we didn't find `" + winNoGPUdll + "`. Trying a GPU run anyway.")
        else:
            self.lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self. predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        if self.hasGPU:
            set_gpu = self.lib.cuda_set_device
            set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int),
                                      c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self. make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self. do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

    def detect_from_image(self, net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im, arr = array_to_image(custom_image)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(net, im)
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
        num = pnum[0]
        if nms:
            self.do_nms_sort(dets, num, meta.classes, nms)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = meta.names[i]
                    else:
                        nameTag = self.altNames[i]
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_detections(dets, num)
        return res

    def detect_from_file(self, net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        im = self.load_image(image, 0, 0)
        if debug: print("Loaded image")
        num = c_int(0)
        if debug: print("Assigned num")
        pnum = pointer(num)
        if debug: print("Assigned pnum")
        self.predict_image(net, im)
        if debug: print("did prediction")
        # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, 0) # OpenCV
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
        if debug: print("Got dets")
        num = pnum[0]
        if debug: print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, meta.classes, nms)
        if debug: print("did sort")
        res = []
        if debug: print("about to range")
        for j in range(num):
            if debug: print("Ranging on " + str(j) + " of " + str(num))
            if debug: print("Classes: " + str(meta), meta.classes, meta.names)
            for i in range(meta.classes):
                if debug: print("Class-ranging on " + str(i) + " of " + str(meta.classes) + "= " + str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = meta.names[i]
                    else:
                        nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug: print("did sort")
        self.free_image(im)
        if debug: print("freed image")
        self.free_detections(dets, num)
        if debug: print("freed detections")
        return res

    def detect(self, image, imageorfile=0, thresh=0.25):
        # 显示图像
        if imageorfile == 0:
            detections = self.detect_from_image(self.netMain, self.metaMain, image, thresh)
        elif imageorfile == 1:
            detections = self.detect_from_file(self.netMain, self.metaMain, image, thresh)
        return detections

    def getNames(self):
        return self.altNames

    def draw_detect_results(self, im, detections, color=(0, 0, 255)):
        for det in detections:
            class_name = det[0]  # 目标类别名称
            prob = det[1]  # 目标概率
            cx = int(det[2][0])  # 目标中心点横坐标
            cy = int(det[2][1])  # 目标中心点纵坐标
            w = int(det[2][2])  # 目标宽度
            h = int(det[2][3])  # 目标高度
            l = cx - w // 2
            t = cy - h // 2
            cv2.rectangle(im, (l, t), (l + w, t + h), color, thickness=1)
            info = '%s,%.3f' % (class_name, prob)
            t = max(10, t - 2)
            cv2.putText(im, info, (l, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 128, 0), 1)