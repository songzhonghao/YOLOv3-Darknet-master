import xml.etree.ElementTree as ET
import os, argparse, sys, shutil

sets=[('train'), ('val'), ('test')]

def new_folder(wd_in):
    if not os.path.exists(wd_in):
        os.mkdir(wd_in)

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(wd_in, image_id, classes):
    in_file = open('%s/Annotations/%s.xml'%(wd_in, image_id))
    out_file = open('%s/labels/%s.txt'%(wd_in, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def get_classes(wd_in, image_set):
    image_ids = open('%s/ImageSets/Main/%s.txt' % (wd_in, image_set)).read().strip().split()
    classes = []
    for image_id in image_ids:
        in_file = open('%s/Annotations/%s.xml' % (wd_in, image_id))
        tree=ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)

    return classes

def my_convert_voc_to_yolo(wd_in, yolov3_folder):
    classes = get_classes(wd_in, 'train')
    print(classes)
    new_folder("%s" % yolov3_folder)
    new_folder("%s/data" % yolov3_folder)
    new_folder("%s/backup" % yolov3_folder)
    new_folder("%s/cfg" % yolov3_folder)

    for image_set in sets:
        if not os.path.exists('%s/labels/' % (wd_in)):
            os.makedirs('%s/labels/' % (wd_in))
        image_ids = open('%s/ImageSets/Main/%s.txt' % (wd_in, image_set)).read().strip().split()
        list_file = open('%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/JPEGImages/%s.jpg\n' % (wd_in, image_id))
            convert_annotation(wd_in, image_id, classes)
        list_file.close()
        shutil.move('%s.txt' % (image_set), '%s/data/%s.txt' % (yolov3_folder, image_set))

    data_file = open('%s/data/my.data' % yolov3_folder, 'w')
    data_file.write('classes=%d\n' % len(classes))
    data_file.write('train = %s/data/train.txt\n' % yolov3_folder)
    data_file.write('val = %s/data/val.txt\n' % yolov3_folder)
    data_file.write('names = %s/data/my.names\n' % yolov3_folder)
    data_file.write('backup = %s/backup/\n' % yolov3_folder)
    data_file.close()

    names_file = open('%s/data/my.names' % yolov3_folder, 'w')
    for cls in classes:
        names_file.write('%s\n' % cls)
    names_file.close()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert dataset from voc to yolov3')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='VOC folder',
                        default=None, type=str)
    parser.add_argument('--des_folder', dest='des_folder',
                        help='yolov3 folder',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    wd_in = args.src_folder.strip()
    yolov3_folder = args.des_folder.strip()
    my_convert_voc_to_yolo(wd_in, yolov3_folder)
