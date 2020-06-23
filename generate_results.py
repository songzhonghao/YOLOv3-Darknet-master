import os
import cv2
import numpy as np
import argparse, sys
from Yolov3Detector import  YOLOV3_Detector

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

def compute_detections(detector, classes, wd_in, savedir, drawresults, thresh=0.25):
    detect_thresh = thresh  # 检测门限

    # 读取评测样本
    files = os.listdir(wd_in)

    count = 0
    fp = {}

    for cls in classes:
        fp[cls] = open("%s/%s.txt" % (savedir, cls), 'w', encoding='utf-8', errors='ignore')

    for fname in files:
        count = count + 1
        postfix = fname.split('.')[-1].strip()
        if postfix != 'jpg' and postfix != 'jpeg' and postfix != 'bmp' and postfix != 'png' and postfix != 'tif':
            continue

        im = cv2.imread("%s/%s" % (wd_in, fname))
        detections = detector.detect(im, 0, detect_thresh)  # 执行检测
        for det in detections:
            pos = np.round(det[2])
            fp[det[0]].write('%s %f %d %d %d %d\n' % (
            fname.split('.')[0].strip(), det[1], pos[0] - pos[2] // 2, pos[1] - pos[3] // 2, pos[0] + pos[2] // 2,
            pos[1] + pos[3] // 2))

        if drawresults:
            detector.draw_detect_results(im, detections)
            cv2.imwrite(os.path.join(savedir, fname), im)

        print("[%d : %d], D = %d" % (count, len(files), len(detections)))

    for cls in classes:
        fp[cls].close()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Compute mAP of YOLOV3 detector')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='images path',
                        default=None, type=str)
    parser.add_argument('--des_folder', dest='des_folder',
                        help='output path',
                        default=None, type=str)
    parser.add_argument('--draw_results', dest='draw_results',
                        help='draw results',
                        default=None, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    yolo_params = {'cfg': './yolov3_train/cfg/yolov3-voc-test.cfg',
                   'weight': './yolov3_train/backup/yolov3-voc-new_32100.weights',
                   'meta': './yolov3_train/data/ship.data',
                   'detect_thresh': 0.25
                   }

    args = parse_args()
    print(args)
    wd_in = args.src_folder
    savedir = args.des_folder
    drawresults = args.draw_results

    if os.path.exists(savedir):
        for file in os.listdir(savedir):
            os.remove(os.path.join(savedir, file))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    classes = init_detector(yolo_params, bLoaded=False)

    #检测器参数设置
    detector = init_detector(yolo_params, bLoaded=True)
    compute_detections(detector, classes, wd_in, savedir, drawresults, thresh=yolo_params['detect_thresh'])

    print("Finished")
