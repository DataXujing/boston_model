# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow

from Ui_my_main_ui import Ui_MainWindow

from PyQt5.QtCore import *
from PyQt5.QtWidgets import  *
from PyQt5 import *
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn.datasets as datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib



class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
    def model_plot(self, y_true,model_pre):
        '''
        y_true:真实的label
        model_pre: 预测的数据(数据框)
        '''
        cols = model_pre.columns
        plt.style.use("ggplot")
        plt.figure(figsize=(24,24))
        plt.rcParams['font.sans-serif'] = ['FangSong'] 
        plt.rcParams['axes.unicode_minus'] = False 
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.scatter(x=range(len(y_true)),y=y_true,label='true')
            plt.scatter(x=range(len(model_pre[cols[i]])),y=model_pre[cols[i]],label=cols[i])
            plt.title(str(cols[i])+':真实Label Vs 预测Label')
            plt.legend()
    
        plt.savefig("model_plot.png")
    
    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print("加载数据")
        
        boston = datasets.load_boston()
        train = boston.data
        target = boston.target
        
        self.X_train,self.x_test,self.y_train,self.y_true = train_test_split(train,target,test_size=0.2)
    
    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print("模型预测")
        
        # 模型加载
        lr_m = joblib.load("model/LR_model.m")
        rr_m = joblib.load("model/RR_model.m")
        llr_m = joblib.load("model/LLR_model.m")
        knnr_m = joblib.load("model/KNNR_model.m")
        dr_m = joblib.load("model/DR_model.m")
        svmr_m = joblib.load("model/SVMR_model.m")
        
        try:
            y_LR = lr_m.predict(self.x_test)
            y_RR = rr_m.predict(self.x_test)
            y_LLR = llr_m.predict(self.x_test)
            y_KNNR = knnr_m.predict(self.x_test)
            y_DR = dr_m.predict(self.x_test)
            y_SVMR = svmr_m.predict(self.x_test)
            
            
            model_pre = pd.DataFrame({'LinearRegression()':list(y_LR),'Ridge()':list(y_RR),'Lasso()':list(y_LLR), \
            'KNeighborsRegressor()':list(y_KNNR),'DecisionTreeRegressor()':list(y_DR),'SVR()':list(y_SVMR)})
            
            self.model_plot(self.y_true, model_pre)
            self.graphicsView.setStyleSheet("border-image: url(model_plot.png);")

            
        except:
            my_button_w3=QMessageBox.warning(self,"严重警告", '请务必先加载数据然后再点击模型预测！！！', QMessageBox.Ok|QMessageBox.Cancel,  QMessageBox.Ok)  
            

    
    @pyqtSlot()
    def on_action_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print('打开')
        my_button_open = QMessageBox.about(self, '打开', '点击我打开某些文件')
    
    @pyqtSlot()
    def on_action_2_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print('关闭')
        sys.exit(0)
    
    @pyqtSlot()
    def on_action_3_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print('联系我们')
        my_button_con_me = QMessageBox.about(self, '联系我们', '这个位置放的是联系我们的介绍')
    
    @pyqtSlot()
    def on_action_4_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print('关于我们')
        my_button_about_me = QMessageBox.about(self, '关于我们', '这个位置放的是关于我们的介绍')
        
    
    @pyqtSlot()
    def on_action_QT_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print('关于qt')
        my_button_about_QT = QMessageBox.aboutQt(self, '关于QT')
        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    splash = QSplashScreen(QtGui.QPixmap(':/my_pic/pic/face.png'))
    splash.show()
    QThread.sleep(0.5)
    splash.showMessage('正在加载机器学习算法...' )
    QThread.sleep(1)
    splash.showMessage('正在初始化程序...')
    QThread.sleep(0.5)
    #splash.show()
    app. processEvents()
    ui =MainWindow()
    # ui.setDaemon(True`)
    # ui.start()
    ui.show()
    splash.finish(ui)
    sys.exit(app.exec_())
    
