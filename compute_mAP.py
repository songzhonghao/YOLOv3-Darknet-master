import xml.etree.ElementTree as ET
import os
import cv2, argparse, sys
import numpy as np
from Yolov3Detector import  YOLOV3_Detector, compute_iou
from  voc_eval import voc_eval

# https://github.com/AlexeyAB/darknet

def init_detector(params, bLoaded=True):
    dllname = "yolo_cpp_dll"
    cfgPath = params['cfg']  # 模型配置文件
    weightPath = params['weight']  # 模型权重
    metaPath = params['meta']# 标签数据

    # 检测器初始化
    detector = YOLOV3_Detector(dllname, cfgPath, weightPath, metaPath)
    if bLoaded:
        detector.loadDLL()
        detector.loadConfig()
    detector.loadNames()
    altNames = detector.getNames()
    if bLoaded:
        return detector
    else:
        return altNames

def compute_detections(detector, classes, wd_in, filename, thresh=0.25, postfix='.jpg'):
    detect_thresh = thresh  # 检测门限
    # 读取评测样本
    fread = open("%s/ImageSets/Main/%s.txt" % (wd_in, filename), 'r', encoding='utf-8', errors='ignore')
    lines = fread.readlines()
    fread.close()

    count = 0
    fp = {}

    for cls in classes:
        fp[cls] = open("%s/%s.txt" % ('results', cls), 'w', encoding='utf-8', errors='ignore')

    for line in lines:
        count = count + 1
        imgfile = "%s/JPEGImages/%s%s" % (wd_in, line.strip(), postfix)
        img = cv2.imread(imgfile)
        detections = detector.detect(img, 0, detect_thresh)  # 执行检测
        for det in detections:
            pos = np.round(det[2])
            fp[det[0]].write('%s %f %d %d %d %d\n' % (
            line.strip(), det[1], pos[0] - pos[2] // 2, pos[1] - pos[3] // 2, pos[0] + pos[2] // 2,
            pos[1] + pos[3] // 2))

        print("[%d,%d], D = %d" % (count, len(lines), len(detections)))

    for cls in classes:
        fp[cls].close()

def my_compute_map(wd_in, yolov3_folder, thresh, filename, postfix):
    yolo_params = {'cfg': '%s/cfg/yolov3-voc-test.cfg' % yolov3_folder,
                   'weight': '%s/backup/yolov3-voc-new_last.weights' % yolov3_folder,
                   'meta': '%s/data/my.data' % yolov3_folder,
                   'detect_thresh': thresh
                   }
    savedir = './results'

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    for file in os.listdir(savedir):
        os.remove(os.path.join(savedir, file))

    classes = init_detector(yolo_params, bLoaded=False)
    if not os.path.exists("%s/%s.txt" % (savedir, classes[0])):
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        # 检测器参数设置
        detector = init_detector(yolo_params, bLoaded=True)
        compute_detections(detector, classes, wd_in, filename, thresh=yolo_params['detect_thresh'], postfix=postfix)

    all_ap = np.zeros((len(classes), 3), np.float)
    for i, cls in enumerate(classes):
        rec, prec, ap = voc_eval('%s/{:s}.txt' % savedir, '%s/Annotations/{:s}.xml' % wd_in,
                                 '%s/ImageSets/Main/%s.txt' % (wd_in, filename), cls, savedir, use_07_metric=True)
        all_ap[i, 0] = ap

    mAP = np.mean(all_ap[:, 0])
    return mAP

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='computation of mAP for yolov3 detector')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='VOC folder',
                        default=None, type=str)
    parser.add_argument('--des_folder', dest='des_folder',
                        help='yolov3 folder',
                        default=None, type=str)
    parser.add_argument('--thresh', dest='thresh',
                        help='0.25',
                        default=0.25, type=float)
    parser.add_argument('--eval_type', dest='eval_type',
                        help='train, eval, test',
                        default='test', type=str)
    parser.add_argument('--postfix', dest='postfix',
                        help='',
                        default='.jpg', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    wd_in = args.src_folder
    filename = args.eval_type
    yolov3_folder = args.des_folder
    thresh = args.thresh
    postfix = args.postfix
    mAP = my_compute_map(wd_in, yolov3_folder, thresh, filename, postfix)
    print("mAP = %f" % mAP)
