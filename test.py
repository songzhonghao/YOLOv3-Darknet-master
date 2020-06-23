import xml.etree.ElementTree as ET
import cv2, sys, argparse
from Yolov3Detector import YOLOV3_Detector

colors = [(255,0,0,), (0,0,255)]

def read_xmlfile(in_file, classes):
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    objinfo = []
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        w = xmax-xmin+1
        h = ymax-ymin+1
        b = [xmin+w/2, ymin+h/2, w, h]
        objinfo.append((b, cls_id))
    return objinfo

def _iou(box1, box2):
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

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def draw_detect_results(image, detections, altNames=None, color=(0,0,255)):
    for det in detections:
        class_name = det[0]  # 目标类别名称
        prob = det[1]  # 目标概率
        cx = int(det[2][0])  # 目标中心点横坐标
        cy = int(det[2][1])  # 目标中心点纵坐标
        w = int(det[2][2])  # 目标宽度
        h = int(det[2][3])  # 目标高度
        l = cx - w // 2
        t = cy - h // 2
        index = altNames.index(class_name)
        cv2.rectangle(image, (l, t), (l+w, t+h), color, thickness=1)
        info = '%s,%.3f' % (class_name, prob)
        t = max(10, t-2)
        cv2.putText(image, info, (l, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,128,0), 1)

def my_test_images(wd_in, yolov3_folder, detect_thresh, val_type, postfix):
    dllname = "yolo_cpp_dll"
    cfgPath = '%s/cfg/yolov3-voc-test.cfg' % yolov3_folder  # 模型配置文件
    weightPath = '%s/backup/yolov3-voc-new_last.weights' % yolov3_folder  # 模型权重
    metaPath = '%s/data/my.data' % yolov3_folder  # 标签数据
    detector = YOLOV3_Detector(dllname, cfgPath, weightPath, metaPath)
    detector.loadDLL()
    detector.loadConfig()
    detector.loadNames()
    altNames = detector.getNames()

    fread = open("%s/ImageSets/Main/%s.txt" % (wd_in, val_type), 'r', encoding='utf-8', errors='ignore')
    lines = fread.readlines()
    count = 0
    cv2.namedWindow("results")
    cv2.moveWindow("results", 0, 0)
    for line in lines:
        count = count + 1
        xmlfile = "%s/%s/%s.xml" % (wd_in, "Annotations", line.strip())
        imgfile = "%s/%s/%s%s" % (wd_in, 'JPEGImages', line.strip(), postfix)
        objinfo = read_xmlfile(xmlfile, altNames)
        im = cv2.imread(imgfile)
        truth = [(altNames[obj[1]], 1, obj[0]) for obj in objinfo]
        detections = detector.detect(im, 0, detect_thresh)  # 执行检测
        print(detections)
        draw_detect_results(im, detections, altNames, (255, 0, 0))
        draw_detect_results(im, truth, altNames)
        cv2.imshow("results", im)
        cv2.waitKey(0)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Split dataset into train and val')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='VOC folder',
                        default=None, type=str)
    parser.add_argument('--des_folder', dest='des_folder',
                        help='yolov3 folder',
                        default=None, type=str)
    parser.add_argument('--thresh', dest='thresh',
                        help='',
                        default=0.25, type=float)
    parser.add_argument('--val_type', dest='val_type',
                        help='',
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
    yolov3_folder = args.des_folder
    detect_thresh = args.thresh  # 检测门限
    val_type = args.val_type
    postfix = args.postfix
    my_test_images(wd_in, yolov3_folder, detect_thresh, val_type, postfix)


