from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import numpy as np
import math
import cv2

np.set_printoptions(threshold=sys.maxsize)

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        #Initial Variables
        self.title = 'CS5550 - Digital Image Processing'
        self.left = 10
        self.top = 10
        self.width = 1440
        self.height = 720
        
        #Variables for image
        self.original_image_path = None
        self.original_image = None
        self.modified_image = None
      
        #For options
        self.bit_modify = 0

        #Starter Functions
        self.initUI()

        self.spinbox = None

        self.groupBoxOriginalImage = QtWidgets.QGroupBox("Original Image", self)
        self.groupBoxOriginalImage.setMinimumSize(512, 512)       
        self.groupBoxModifiedImage = QtWidgets.QGroupBox("Modified Image", self)
        self.groupBoxModifiedImage.setMinimumSize(512, 512)
        self.groupBoxModifiedImage.move(512, 0)
        self.groupBoxOptions = QtWidgets.QGroupBox("Options", self)
        self.groupBoxOptions.move(1024, 0)

        layout_original_image = QtWidgets.QHBoxLayout()
        self.label_original_image = QtWidgets.QLabel()
        layout_original_image.addWidget(self.label_original_image)
        self.groupBoxOriginalImage.setLayout(layout_original_image)
        
        layout_modified_image = QtWidgets.QHBoxLayout()
        self.label_modified_image = QtWidgets.QLabel()
        layout_modified_image.addWidget(self.label_modified_image)
        self.groupBoxModifiedImage.setLayout(layout_modified_image)

        #Show application
        self.show()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #Set the main menu
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        editMenu = mainMenu.addMenu('Edit')

        #File menu options
        loadFile = QtWidgets.QAction("Load Image", self)
        loadFile.triggered.connect(self.load_file)
        fileMenu.addAction(loadFile)

        #Edit menu options
        zoomEdit = QtWidgets.QAction("Zoom/Shrink Image", self)
        gray_scale_resEdit = QtWidgets.QAction("Resize Gray Scale Resolution", self)
        zoomEdit.triggered.connect(self.zoom_dbox)
        gray_scale_resEdit.triggered.connect(self.gray_scale_res_dbox)
        editMenu.addAction(zoomEdit)
        editMenu.addAction(gray_scale_resEdit)
        

    def load_file(self):
        self.original_image_path = QtWidgets.QFileDialog.getOpenFileName(self, 'OpenFile')[0]
        if self.original_image_path is not None:
            pixmap = QtGui.QPixmap(self.original_image_path)
            self.original_image = cv2.imread(self.original_image_path, cv2.IMREAD_GRAYSCALE)
            self.label_original_image.setPixmap(pixmap)
    
    def zoom_dbox(self):
        dialogBox = QtWidgets.QDialog()
        dialogBox.setMinimumSize(500, 300)
        dialogBox.setMaximumSize(500, 300)
        dialogBox.setWindowTitle("Zoom/Shrink")
        dialogBox.move(50, 50)
        layout_dbox = QtWidgets.QVBoxLayout()
        radioBoxNN = QtWidgets.QRadioButton("Nearest Neighbor Interpolation")
        radioBoxLI = QtWidgets.QRadioButton("Linear Interpolation")
        radioBoxBI = QtWidgets.QRadioButton("Bilinear Interpolation")
        inputH = QtWidgets.QLineEdit()
        inputW = QtWidgets.QLineEdit()
        labelH = QtWidgets.QLabel("Height (px)")
        labelW = QtWidgets.QLabel("Width (px)")
        button = QtWidgets.QPushButton("OK")
        layout_dbox.addWidget(radioBoxNN)
        layout_dbox.addWidget(radioBoxLI)
        layout_dbox.addWidget(radioBoxBI)
        layout_dbox.addWidget(labelH)
        layout_dbox.addWidget(inputH)
        layout_dbox.addWidget(labelW)
        layout_dbox.addWidget(inputW)
        layout_dbox.addWidget(button)

        dialogBox.setLayout(layout_dbox)
        dialogBox.exec_()

    def gray_scale_res_dbox(self):
        dialogBox = QtWidgets.QDialog()
        dialogBox.setMinimumSize(500, 100)
        dialogBox.setMaximumSize(500, 100)
        dialogBox.setWindowTitle("Gray Scale Resolution")
        dialogBox.move(50, 50)
        layout_dbox = QtWidgets.QHBoxLayout()
        dialogText = QtWidgets.QLabel('Gray Level Resolution in Bits: ')
        layout_dbox.addWidget(dialogText)

        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setRange(0, 8)
        self.spinbox.setValue(8)
        layout_dbox.addWidget(self.spinbox)
        button = QtWidgets.QPushButton('OK')
        layout_dbox.addWidget(button)
        button.clicked.connect(self.bit_modify_image)
        dialogBox.setLayout(layout_dbox)
        dialogBox.exec_()
    
    def bit_modify_image(self):
        sh = self.original_image.shape
        self.bit_modify = self.spinbox.value()
        self.modified_image = np.zeros(sh, dtype=np.uint8)
        #print(self.original_image)
        for i in range(self.original_image.shape[0]):
            for j in range(self.original_image.shape[1]):
                ovalue = self.original_image[i, j]
                #print('original value:' )
                #print(ovalue)
                ovalue = float(ovalue) / (2**8 - 1)
                #print('normalized:')
                #print(ovalue)
                ovalue = (2**self.bit_modify - 1) * ovalue
                ovalue = round(ovalue)
                #print('bit value:')
                #print(ovalue)
                ovalue = ovalue / (2**self.bit_modify - 1)
                #print('renormalized:')
                #print(ovalue)
                ovalue = round(ovalue * (2**8 - 1))
                #print('rounded:')
                #print(ovalue)
                ovalue = int(ovalue)
                #print('modified value:')
                #print(ovalue)
                self.modified_image[i, j] = ovalue
                #print('---')
        print(np.amax(self.modified_image))
        image = self.numpy_arr_to_pixmap()
        self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image))

    def numpy_arr_to_pixmap(self):
        sh = self.modified_image.shape
        image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8) 
        return image

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
