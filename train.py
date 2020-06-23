import os, argparse, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Split dataset into train and val')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='VOC folder',
                        default=None, type=str)
    parser.add_argument('--train_prop', dest='train_prop',
                        help='propotion of train set, 0.5',
                        default=0.9, type=float)
    parser.add_argument('--val_prop', dest='val_prop',
                        help='propotion of val set, 0.3',
                        default=0.05, type=float)
    parser.add_argument('--des_folder', dest='des_folder',
                        help='yolov3 folder',
                        default=None, type=str)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='batch size, 1-4',
                        default=4, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    voc_in = args.src_folder
    train_prop = args.train_prop
    val_prop = args.val_prop
    yolov3_out = args.des_folder
    batch_size = args.batch_size
    print("split dataset into train, val and test\n")
    os.system(r"python split_data.py --src_folder %s --train_prop %f --val_prop %f" % \
              (voc_in, train_prop, val_prop))
    print("convert dataset format from voc to yolov3")
    os.system(r"python voc_label.py --src_folder %s --des_folder %s" % (voc_in, yolov3_out))
    print("convert configuration file of yolov3")
    os.system(r"python convert_cfg.py --src_folder %s --batch_size %d" % (yolov3_out, batch_size))
    print("train started")
    os.system(r"darknet.exe detector train %s/data/my.data %s/cfg/yolov3-voc-new.cfg" % (yolov3_out, yolov3_out))
