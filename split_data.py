import os
import math
import random
import argparse, sys

def new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def my_split_dataset(root_path, train_prop, val_prop):
    dataset_path = '%s/JPEGImages/' % root_path
    new_path = '%s/ImageSets/Main/' % root_path
    new_folder(new_path)

    train_path = new_path + 'train.txt'
    train_flags = False
    if os.path.exists(train_path):
        train_flags = True
    else:
        ftrain = open(train_path, 'w', encoding='utf-8', errors='ignore')

    test_path = new_path + 'test.txt'

    test_flags = False
    if os.path.exists(test_path):
        test_flags = True
    else:
        ftest = open(test_path, 'w', encoding='utf-8', errors='ignore')

    valid_path = new_path + 'val.txt'

    valid_flags = False
    if os.path.exists(valid_path):
        valid_flags = True
    else:
        fvalid = open(valid_path, 'w', encoding='utf-8', errors='ignore')

    file_list = os.listdir(dataset_path)
    random.shuffle(file_list)
    file_count = len(file_list)
    train_count = math.floor(file_count * train_prop)
    valid_count = math.floor(file_count * (train_prop + val_prop))

    count = 0

    for f in file_list:
        fname = f.strip().split('.')[0]
        path = '%s/Annotations/%s.xml' % (root_path, fname)
        if not os.path.exists(path):
            print(path)
            continue
        if count < train_count:
            if train_flags == False:
                ftrain.write(fname + '\n')
        elif count < valid_count:
            if valid_flags == False:
                fvalid.write(fname + '\n')
        else:
            if test_flags == False:
                ftest.write(fname + '\n')
        count = count + 1

    if train_flags == False:
        ftrain.close()
    if test_flags == False:
        ftest.close()
    if valid_flags == False:
        fvalid.close()

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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    root_path = args.src_folder.strip()
    train_prop = args.train_prop
    val_prop = args.val_prop
    my_split_dataset(root_path, train_prop, val_prop)