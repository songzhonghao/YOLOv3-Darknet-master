import os
from main import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from split_data import my_split_dataset
from voc_label import my_convert_voc_to_yolo
from convert_cfg import my_generate_cfgfile
from test import my_test_images
from compute_mAP import my_compute_map

class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def  __init__ (self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.train_allstep.clicked.connect(self.train)
        self.vocSetup.clicked.connect(self.open_vocpath)
        self.yoloSetup.clicked.connect(self.open_yolopath)
        self.trainsetSetup.clicked.connect(self.split_dataset)
        self.format_convert.clicked.connect(self.voc_formatconvert)
        self.config_generate.clicked.connect(self.configfile_genenrate)
        self.train_yolo.clicked.connect(self.train_yolov3)
        self.test_oneimage.clicked.connect(self.test_image)
        self.compute_mAP.clicked.connect(self.compute_mymAP)
        self.yolov3_templatemodel = ['template3.cfg', 'template4.cfg', 'template5.cfg']

    def show_progress(self, txt):
        self.statusbar.showMessage(txt)

    def check_args(self):
        try:
            self.voc_in = self.voc_path.text()
            self.train_prop = float(self.train_coef.text())
            self.val_prop = float(self.val_coef.text())
            self.yolov3_out = self.yolo_path.text()
            self.batch_size = int(self.batchsize.value())
            self.thresh = float(self.test_thresh.text())
            self.testtype = self.test_type.currentText()
            self.postfix = self.image_postfix.currentText()
            self.yolov3model = self.yolov3_templatemodel[self.modelType.currentIndex()]
            return True
        except:
            box = QMessageBox()
            box.setWindowTitle("提示")
            box.setText("参数设置错误")
            box.exec()
            return False
        return True

    def test_image(self):
        if self.check_args() == False:
            return
        # os.system(r"python test.py --src_folder %s  --des_folder %s --thresh %f --val_type %s --postfix %s" %  \
        #           (self.voc_path, self.yolov3_out, self.thresh, self.testtype, self.postfix))
        self.show_progress(r"测试图像开始")
        my_test_images(self.voc_path, self.yolov3_out, self.thresh, self.testtype, self.postfix)

    def compute_mymAP(self):
        if self.check_args() == False:
            return
        # os.system(r"python compute_mAP.py --src_folder %s --des_folder %s --eval_type %s --thresh %f --postfix %s" % \
        #           (self.voc_path, self.yolov3_out, self.thresh, self.testtype, self.postfix))
        mAP=my_compute_map(self.voc_path, self.yolov3_out, self.thresh, self.testtype, self.postfix)
        self.show_progress("mAP= %f" % mAP)

    def configfile_genenrate(self):
        if self.check_args() == False:
            return

        print("convert configuration file of yolov3")
        # os.system(r"python convert_cfg.py --src_folder %s --batch_size %d" % (self.yolov3_out, self.batch_size))
        my_generate_cfgfile(self.yolov3_out, self.batch_size, self.yolov3model)
        self.show_progress(r"已生成YOLO配置文件")

    def train_yolov3(self):
        if self.check_args() == False:
            return

        print("train started")
        self.show_progress(r"训练YOLOV3模型开始")
        os.system(r"darknet.exe detector train %s/data/my.data %s/cfg/yolov3-voc-new.cfg" % (self.yolov3_out, self.yolov3_out))

    def voc_formatconvert(self):
        if self.check_args() == False:
            return

        print("convert dataset format from voc to yolov3")
        self.show_progress(r"格式转换开始")
        # os.system(r"python voc_label.py --src_folder %s --des_folder %s" % (self.voc_in, self.yolov3_out))
        my_convert_voc_to_yolo(self.voc_in, self.yolov3_out)
        self.show_progress(r"格式转换结束")

    def split_dataset(self):
        if self.check_args() == False:
            return

        print("split dataset into train, val and test\n")
        # os.system(r"python split_data.py --src_folder %s --train_prop %f --val_prop %f" % \
        #           (self.voc_in, self.train_prop, self.val_prop))
        self.show_progress(r"切分VOC数据集开始")
        my_split_dataset(self.voc_in, self.train_prop, self.val_prop)
        self.show_progress(r"切分VOC数据集结束")

    def open_vocpath(self):
        fname = QFileDialog.getExistingDirectory(self, 'VOC数据集路径', '.')
        if fname:
            self.voc_path.setText(fname)
            self.show_progress(r"设置VOC路径完成")
        else:
            box = QMessageBox()
            box.setWindowTitle("提示")
            box.setText("先设置VOC数据集路径")
            box.exec()
            return

    def open_yolopath(self):
        fname = QFileDialog.getExistingDirectory(self, 'YOLO配置文件保存文件夹', '.')
        if fname:
            self.yolo_path.setText(fname)
            self.show_progress(r"设置YOLO配置文件路径完成")
        else:
            box = QMessageBox()
            box.setWindowTitle("提示")
            box.setText("先设置YOLO配置文件保存路径")
            box.exec()
            return

    def stop(self):
        os.system('taskkill /IM darknet.exe /F')
        self.show_progress('训练结束')

    def train(self):
        if self.check_args() == False:
            return

        # print("split dataset into train, val and test\n")
        # os.system(r"python split_data.py --src_folder %s --train_prop %f --val_prop %f" % \
        #           (self.voc_in, self.train_prop, self.val_prop))
        # print("convert dataset format from voc to yolov3")
        # os.system(r"python voc_label.py --src_folder %s --des_folder %s" % (self.voc_in, self.yolov3_out))
        # print("convert configuration file of yolov3")
        # os.system(r"python convert_cfg.py --src_folder %s --batch_size %d" % (self.yolov3_out, self.batch_size))
        # print("train started")
        my_split_dataset(self.voc_in, self.train_prop, self.val_prop)
        my_convert_voc_to_yolo(self.voc_in, self.yolov3_out)
        my_generate_cfgfile(self.yolov3_out, self.batch_size, self.yolov3model)
        os.system(r"darknet.exe detector train %s/data/my.data %s/cfg/yolov3-voc-new.cfg" % (self.yolov3_out, self.yolov3_out))