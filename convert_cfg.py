import os, argparse, sys

def my_generate_cfgfile(yolov3_folder, batch_size, cfg_name = 'template.cfg'):
    classes = open("%s/data/my.names" % yolov3_folder).read().strip().split()
    cfg_files = open(cfg_name)
    lines = cfg_files.readlines()
    cfg_files.close()
    params = []
    for index, line in enumerate(lines):
        if "subdivisions" in line:
            part = line.strip().split('=')
            params.append((index, int(part[-1].strip())))
        if "anchors" in line:
            part = line.strip().split('=')[-1].strip().split(',')
            params.append((index, len(part) // 2))

    print(params)

    for index, line in enumerate(lines):
        if "batch=" in line:
            lines[index] = "batch=%d\n" % (batch_size * params[0][1])
            break

    paramsLen = len(lines) - 1
    findFlags = False
    for index in range(len(lines)):
        if "classes" in lines[paramsLen - index]:
            lines[paramsLen - index] = "classes = %d\n" % len(classes)
        if "yolo" in lines[paramsLen - index]:
            findFlags = True
        if "filters" in lines[paramsLen - index] and findFlags == True:
            lines[paramsLen - index] = "filters = %d\n" % (3 * (4 + 1 + len(classes)))
            findFlags = False

    save_name = "%s/cfg/yolov3-voc-new.cfg" % yolov3_folder
    cfg_files = open(save_name, 'w')
    for line in lines:
        cfg_files.write(line)
    cfg_files.close()

    save_name = "%s/cfg/yolov3-voc-test.cfg" % yolov3_folder
    cfg_files = open(save_name, 'w')
    for line in lines:
        if "batch=" in line:
            line = "batch=1\n"
        elif "subdivisions=" in line:
            line = "subdivisions=1\n"
        cfg_files.write(line)
    cfg_files.close()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert configuration file of yolov3')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='yolov3 src folder',
                        default='./yolov3_train', type=str)
    parser.add_argument('--cfg_file', dest='cfg_file',
                        help='cfg path',
                        default='template4.cfg', type=str)
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
    yolov3_folder = args.src_folder
    cfg_name = args.cfg_file.strip()
    batch_size = args.batch_size
    my_generate_cfgfile(yolov3_folder, batch_size, cfg_name)



