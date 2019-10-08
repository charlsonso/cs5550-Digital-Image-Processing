from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import numpy as np
import math
import cv2
import skimage

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
        self.zoom_shrink_option = 0
        self.bit_plane = [False]*8
        self.he_option = 0
        self.he_local_mask_size = 0
        self.filter_option = 0
        self.kernel = 0
        self.highboost = 0

        #Starter Functions
        self.initUI()

        self.spinbox = None

        self.groupBoxOriginalImage = QtWidgets.QGroupBox("Original Image", self)
        self.groupBoxOriginalImage.setMinimumSize(512, 512)       
        self.groupBoxModifiedImage = QtWidgets.QGroupBox("Modified Image", self)
        self.groupBoxModifiedImage.setMinimumSize(512, 512)
        self.groupBoxModifiedImage.move(512, 0)
        self.groupBoxOptions = QtWidgets.QGroupBox("Options", self)
        self.groupBoxOptions.setMinimumSize(300, 300)
        self.groupBoxOptions.move(1024, 0)

        layout_original_image = QtWidgets.QHBoxLayout()
        self.label_original_image = QtWidgets.QLabel()
        layout_original_image.addWidget(self.label_original_image)
        self.groupBoxOriginalImage.setLayout(layout_original_image)
        
        layout_modified_image = QtWidgets.QHBoxLayout()
        self.label_modified_image = QtWidgets.QLabel()
        layout_modified_image.addWidget(self.label_modified_image)
        self.groupBoxModifiedImage.setLayout(layout_modified_image)

        self.optionButton = QtWidgets.QPushButton('Load Modified Image')
        layout_options = QtWidgets.QHBoxLayout()
        layout_options.addWidget(self.optionButton)
        self.groupBoxOptions.setLayout(layout_options)
        self.optionButton.clicked.connect(self.load_modified_image)

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

        histogram_eq = QtWidgets.QAction("Histogram Equalization", self)
        bit_plane = QtWidgets.QAction("Bit Plane Slicing", self)
        filter_option = QtWidgets.QAction("Filter", self)

        zoomEdit.triggered.connect(self.zoom_dbox)
        gray_scale_resEdit.triggered.connect(self.gray_scale_res_dbox)
        histogram_eq.triggered.connect(self.histogram_dbox)
        filter_option.triggered.connect(self.filter_dbox)
        bit_plane.triggered.connect(self.bit_plane_dbox)

        editMenu.addAction(zoomEdit)
        editMenu.addAction(gray_scale_resEdit)
        editMenu.addAction(histogram_eq)
        editMenu.addAction(filter_option)
        editMenu.addAction(bit_plane)
    
    def filter_dbox(self):
        dbox = self.create_dbox('Filter Options')
        layout_dbox = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel('Kernel Size')
        self.kernel = QtWidgets.QLineEdit()
        smoothing_filter = QtWidgets.QRadioButton('Smoothing Filter')
        smoothing_filter.option = 1
        smoothing_filter.toggled.connect(self.change_filter_option)
        median_filter = QtWidgets.QRadioButton('Median Filter')
        median_filter.option = 2
        median_filter.toggled.connect(self.change_filter_option)
        laplacian_filter = QtWidgets.QRadioButton('Sharpening Laplacian Filter')
        laplacian_filter.option = 3
        laplacian_filter.toggled.connect(self.change_filter_option)
        high_boost_filter = QtWidgets.QRadioButton('High Boosting Filter')
        high_boost_filter.option = 4
        high_boost_filter.toggled.connect(self.change_filter_option)

        label_high_boost = QtWidgets.QLabel('High Boost A Value:')
        self.highboost = QtWidgets.QLineEdit()

        button = QtWidgets.QPushButton("OK")
        layout_dbox.addWidget(label)
        layout_dbox.addWidget(self.kernel)
        layout_dbox.addWidget(smoothing_filter)
        layout_dbox.addWidget(median_filter)
        layout_dbox.addWidget(laplacian_filter)
        layout_dbox.addWidget(high_boost_filter)
        layout_dbox.addWidget(label_high_boost)
        layout_dbox.addWidget(self.highboost)
        layout_dbox.addWidget(button)
        
        button.clicked.connect(self.apply_filter)
        dbox.setLayout(layout_dbox)
        dbox.exec_()

    def change_filter_option(self):
        button = self.sender()
        option = button.option
        if option != 0:
            self.filter_option = option
    
    def apply_filter(self):
        kernel_size = int(self.kernel.text())
        highboost = int(self.highboost.text())
        if self.filter_option == 1:
            self.apply_smoothing_filter(kernel_size)
        elif self.filter_option == 2:
            self.apply_median_filter(kernel_size)
        elif self.filter_option == 3:
            self.apply_laplacian_filter(kernel_size)
        elif self.filter_option == 4:
            self.apply_highboost_filter(kernel_size, highboost)

    def apply_highboost_filter(self, kernel_size, highboost):
        kernel = np.ones((kernel_size, kernel_size), dtype = 'uint8')/(kernel_size**2)
        smoothed = cv2.filter2D(self.original_image, -1, kernel)
        mask = self.original_image - smoothed
        self.modified_image = mask*highboost + self.original_image
        self.set_modified_image()

    def apply_laplacian_filter(self, kernel_size):
        pad = int(kernel_size/2)
        padded_image = np.pad(self.original_image, (pad, pad), 'constant', constant_values=(0))
        self.modified_image = padded_image
        sh = self.original_image.shape
        self.modified_image = np.zeros(self.original_image.shape, dtype='uint8')
        laplace_kernel = np.ones((kernel_size, kernel_size), dtype = int)
        val = 1 - kernel_size**2
        laplace_kernel[pad, pad] = val 
        print('lplace_kernel')
        print(laplace_kernel)
        for w in range(sh[1]):
            for h in range(sh[0]):
                new_h = pad + h
                new_w = pad + w
                start_h = new_h - pad
                start_w = new_w - pad
                end_h = new_h + pad + 1
                end_w = new_w + pad + 1
                local_image_matrix = padded_image[start_h:end_h, start_w:end_w]
                matrix = laplace_kernel * local_image_matrix
                total = matrix.sum() * -1
                self.modified_image[h, w] = total + self.original_image[h,w] 
        self.set_modified_image()

    def apply_median_filter(self, kernel_size):
        pad = int(kernel_size/2)
        padded_image = np.pad(self.original_image, (pad, pad), 'constant', constant_values=(0))
        sh = self.original_image.shape
        self.modified_image = np.zeros(self.original_image.shape, dtype = 'uint8')
        for w in range(sh[1]):
            for h in range(sh[0]):
                new_h = pad + h
                new_w = pad + w
                start_h = new_h - pad
                start_w = new_w - pad
                end_h = new_h + pad + 1
                end_w = new_w + pad + 1
                local_image_matrix = padded_image[start_h:end_h, start_w:end_w] 
                self.modified_image[h, w] = np.median(local_image_matrix) 
        self.modified_image = self.modified_image.astype('uint8')
        self.set_modified_image()

    def apply_smoothing_filter(self, kernel_size):
        pad = int(kernel_size/2)
        padded_image = np.pad(self.original_image, (pad, pad), 'constant', constant_values=(0))
        smoothing_filter = np.ones((kernel_size, kernel_size))  * (1/ (kernel_size * kernel_size))
        sh = self.original_image.shape
        self.modified_image = np.zeros(self.original_image.shape, dtype = 'uint8')
        for w in range(sh[1]):
            for h in range(sh[0]):
                new_h = pad + h
                new_w = pad + w
                start_h = new_h - pad
                start_w = new_w - pad
                end_h = new_h + pad + 1
                end_w = new_w + pad + 1
                local_image_matrix = padded_image[start_h:end_h, start_w:end_w] 
                convoluted = local_image_matrix * smoothing_filter
                self.modified_image[h, w] = convoluted.sum(dtype='uint8')
        self.set_modified_image()

    def histogram_dbox(self):
        dbox = self.create_dbox('Histogram Equalization')
        layout_dbox = QtWidgets.QVBoxLayout()
        global_he = QtWidgets.QRadioButton('Global Equalization')
        global_he.option = 1
        global_he.toggled.connect(self.change_he_option)
        local_he = QtWidgets.QRadioButton('Local Equalization')
        local_he.option = 2
        local_he.toggled.connect(self.change_he_option)
        self.kernel = QtWidgets.QLineEdit()
        labelH = QtWidgets.QLabel("Kernel Size for Local Histogram Equalization (px)")
       

        button = QtWidgets.QPushButton("OK")
        layout_dbox.addWidget(global_he)
        layout_dbox.addWidget(local_he)
        layout_dbox.addWidget(labelH)
        layout_dbox.addWidget(self.kernel)
        layout_dbox.addWidget(button)
        
        button.clicked.connect(self.apply_he)

        dbox.setLayout(layout_dbox)
        dbox.exec_()
   
    def change_he_option(self):
        radio = self.sender()
        option = radio.option
        if radio.isChecked():
            self.he_option = option

    def apply_he(self):
        if self.he_option == 1:
            self.apply_global_he()
        elif self.he_option == 2:
            self.apply_local_he(int(self.kernel.text()))

    def apply_local_he(self, kernel_size):
        img_rescale = skimage.exposure.equalize_hist(self.original_image)
        selem = np.ones((kernel_size, kernel_size)) 
        img_eq = skimage.filters.rank.equalize(self.original_image, selem = selem)
        sh = self.original_image.shape
        self.modified_image = img_eq
        image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8) 
        self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image))

    def apply_global_he(self):
        flat = self.original_image.flatten()
        histogram = np.zeros(256)
        for pixel in flat:
            histogram[pixel] += 1
        a = iter(histogram)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        cumulative = np.array(b)

        nj = (cumulative - cumulative.min())*255
        N = cumulative.max() - cumulative.min()
        cumulative = nj / N
        cumulative = cumulative.astype('uint8')
        img_new = cumulative[flat]
        img_new = np.reshape(img_new, self.original_image.shape)
        self.modified_image = img_new
        sh = self.modified_image.shape
        image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8) 
        self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image))

    def create_dbox(self, name):
        dialogBox = QtWidgets.QDialog()
        dialogBox.setMinimumSize(500, 300)
        dialogBox.setMaximumSize(500, 300)
        dialogBox.setWindowTitle(name)
        dialogBox.move(50, 50)
        return dialogBox

    def bit_plane_dbox(self):
        dbox = self.create_dbox('Bit Plane Slicing')
        layout_dbox = QtWidgets.QHBoxLayout()
        
        check_box0 = QtWidgets.QCheckBox('0', self)
        check_box1 = QtWidgets.QCheckBox('1', self)
        check_box2 = QtWidgets.QCheckBox('2', self)
        check_box3 = QtWidgets.QCheckBox('3', self)
        check_box4 = QtWidgets.QCheckBox('4', self)
        check_box5 = QtWidgets.QCheckBox('5', self)
        check_box6 = QtWidgets.QCheckBox('6', self)
        check_box7 = QtWidgets.QCheckBox('7', self)

        check_box0.option = 0
        check_box1.option = 1
        check_box2.option = 2
        check_box3.option = 3
        check_box4.option = 4
        check_box5.option = 5
        check_box6.option = 6
        check_box7.option = 7

        button = QtWidgets.QPushButton("OK")

        layout_dbox.addWidget(check_box0)
        layout_dbox.addWidget(check_box1)
        layout_dbox.addWidget(check_box2)
        layout_dbox.addWidget(check_box3)
        layout_dbox.addWidget(check_box4)
        layout_dbox.addWidget(check_box5)
        layout_dbox.addWidget(check_box6)
        layout_dbox.addWidget(check_box7)
        layout_dbox.addWidget(button)

        check_box0.stateChanged.connect(self.set_bit_plane) 
        check_box1.stateChanged.connect(self.set_bit_plane) 
        check_box2.stateChanged.connect(self.set_bit_plane) 
        check_box3.stateChanged.connect(self.set_bit_plane) 
        check_box4.stateChanged.connect(self.set_bit_plane) 
        check_box5.stateChanged.connect(self.set_bit_plane) 
        check_box6.stateChanged.connect(self.set_bit_plane) 
        check_box7.stateChanged.connect(self.set_bit_plane) 

        button.clicked.connect(self.adjust_bit_plane_image)

        dbox.setLayout(layout_dbox)
        dbox.exec_()

    def set_bit_plane(self):
        check_button = self.sender()
        option = check_button.option
        if check_button.isChecked():
            self.bit_plane[option] = True
        else:
            self.bit_plane[option] = False

    def adjust_bit_plane_image(self):
        mask = 0 
        for idx, i in enumerate(self.bit_plane):
            if i is False:
                mask = mask | 2**idx
        print(bin(mask))
        sh = self.original_image.shape
        self.modified_image = np.zeros(sh, dtype=np.uint8)
        for w in range(sh[1]):
            for h in range(sh[0]):
                val = self.original_image[w, h]
                self.modified_image[w,h] = val & mask
        sh = self.modified_image.shape
        image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8) 
        self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image))

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
        radioBoxNN.option = 1
        radioBoxLI = QtWidgets.QRadioButton("Linear Interpolation")
        radioBoxLI.option = 2
        radioBoxBI = QtWidgets.QRadioButton("Bilinear Interpolation")
        radioBoxBI.option = 3
        self.inputH = QtWidgets.QLineEdit()
        self.inputW = QtWidgets.QLineEdit()
        labelH = QtWidgets.QLabel("Height (px)")
        labelW = QtWidgets.QLabel("Width (px)")
        button = QtWidgets.QPushButton("OK")
        
        layout_dbox.addWidget(radioBoxNN)
        layout_dbox.addWidget(radioBoxLI)
        layout_dbox.addWidget(radioBoxBI)
        layout_dbox.addWidget(labelH)
        layout_dbox.addWidget(self.inputH)
        layout_dbox.addWidget(labelW)
        layout_dbox.addWidget(self.inputW)
        layout_dbox.addWidget(button)

        radioBoxNN.toggled.connect(self.onRadioClick)
        radioBoxLI.toggled.connect(self.onRadioClick)
        radioBoxBI.toggled.connect(self.onRadioClick)
        button.clicked.connect(self.resize_image)
        dialogBox.setLayout(layout_dbox)
        dialogBox.exec_()

    def onRadioClick(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.zoom_shrink_option = radioButton.option

    def resize_image(self):
        new_height = int(self.inputH.text())
        new_width = int(self.inputW.text())
        self.set_new_modified_image_arr(new_height, new_width)
        #NN
        if self.zoom_shrink_option == 1:
           self.nearest_neighbor() 
        #LI
        elif self.zoom_shrink_option == 2:
            self.linear_interpolation()
        #BI
        elif self.zoom_shrink_option == 3:
            self.bilinear_interpolation()

    def set_new_modified_image_arr(self, h, w):
        self.modified_image = np.zeros((h, w), dtype=np.uint8)

    def nearest_neighbor(self):
         sh = self.modified_image.shape
         osh = self.original_image.shape
         wfactor = sh[1] / osh[1]
         hfactor = sh[0] / osh[0]
         for col in range(sh[1]):
             for row in range(sh[0]):
                 val = self.original_image[int(col/hfactor), int(row/wfactor)]
                 self.modified_image[col, row] = val
         image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8) 
         self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image))

    def linear_interpolation(self):
        sh = self.modified_image.shape
        osh = self.original_image.shape
        for col in range(sh[1]):
             for row in range(sh[0]):
                wfactor = sh[1] / osh[1]
                hfactor = sh[0] / osh[0]
                o_pos_col = (1/wfactor) * col
                o_pos_row = (1/hfactor) * row
                int_col = int(o_pos_col)
                int_row = int(o_pos_row)
                sol_point = []
                sol_point.append((int_row, int_col))
                if (int_col == osh[1] - 1):
                    sol_point.append((int_row, int_col - 1))
                else:
                    sol_point.append((int_row, int_col + 1))
                sm = self.linear_inter_helper(o_pos_col, sol_point[0][1], sol_point[1][1], self.original_image[sol_point[0][1], sol_point[0][0]], self.original_image[sol_point[1][1], sol_point[1][0]])
                self.modified_image[row, col] = sm
        image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8)
        self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image))

    def linear_inter_helper(self, sample_x, close1_x, close2_x, close1_val, close2_val):
        v1 = (close2_x - sample_x)/(close2_x - close1_x) * close1_val
        v2 = (sample_x - close1_x)/(close2_x - close1_x) * close2_val
        return v1 + v2

#    def find_closest_points(self, x, y, num_points):
        

#        list_of_points = []
#        list_of_points.append((math.floor(x), math.floor(y)))
#        list_of_points.append((math.floor(x), math.ceil(y)))
#        list_of_points.append((math.ceil(x), math.floor(y)))
#        list_of_points.append((math.ceil(x), math.ceil(y)))
#        distance_points = []
#        sol_points_idx = []
#        for i in list_of_points:
#            distance_points.append(((x-i[0])**2 + (y-i[1])**2)**(1/2))
#        
#        least_value = 10000
#        index = -1
#        while len(sol_points_idx) < num_points:
#            for idx, i in enumerate(distance_points):
#                if i < least_value:
#                    least_value = i
#                    index = idx
#            sol_points_idx.append(index)
#        sol_point = []
#        for i in sol_points_idx:
#            sol_point.append(list_of_points[i])
#        return sol_point
#
    def bilinear_interpolation(self):
        print('1') 
        bilinear_img = cv2.resize(self.original_image, (32,32), interpolation = cv2.INTER_LINEAR)
        print('2')
        bilinear_zoom_out = cv2.resize(bilinear_img, (480, 480), interpolation=cv2.INTER_LINEAR)
        print('3')
        image = QtGui.QImage(bilinear_img, 32, 32, QtGui.QImage.Format_Grayscale8)
        print('4')
        self.label_original_image.setPixmap(QtGui.QPixmap.fromImage(image))
        print('5')
        image2 = QtGui.QImage(bilinear_zoom_out, 480, 480, QtGui.QImage.Format_Grayscale8)
        print('6')
        self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image2))
 

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
        image = self.numpy_arr_to_pixmap()
        self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image))

    def numpy_arr_to_pixmap(self):
        sh = self.modified_image.shape
        image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8) 
        return image
    
    def load_modified_image(self):
        self.original_image = self.modified_image
        sh = self.modified_image.shape
        image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8) 
        empty = QtGui.QPixmap()
        self.label_original_image.setPixmap(QtGui.QPixmap.fromImage(image))
        self.label_modified_image.clear()

    def set_modified_image(self):
        sh = self.modified_image.shape
        image = QtGui.QImage(self.modified_image, sh[1], sh[0], QtGui.QImage.Format_Grayscale8) 
        self.label_modified_image.setPixmap(QtGui.QPixmap.fromImage(image))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
