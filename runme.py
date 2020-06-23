import sys
from PyQt5 import QtWidgets
from mywindow import MyWindow

if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv) #外部参数列表
    mywin = MyWindow() #我就是要合体的类哦 如果是空的可以直接定义 QtWidgets.QDialog也是可以的，会让新手很迷惑
    mywin.show()      #合体后的成功展示喽
    sys.exit(app.exec_()) #退出中使用的消息循环，结束消息循环时就退出程序