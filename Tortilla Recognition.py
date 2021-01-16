from PyQt5 import QtCore, QtWidgets
import sys
from pathlib import Path
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import numpy as np
import os
import glob
import time

class Ui_Form(QtWidgets.QDialog):
    final_path=""

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(550, 120)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_3.addWidget(self.lineEdit)
        self.browseButton = QtWidgets.QPushButton(Form)
        self.browseButton.setObjectName("browseButton")
        self.horizontalLayout_3.addWidget(self.browseButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.submitButton = QtWidgets.QPushButton(Form)
        self.submitButton.setObjectName("submitButton")
        self.verticalLayout.addWidget(self.submitButton)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Select Video"))
        self.browseButton.setText(_translate("Form", "Browse Video"))
        self.submitButton.setText(_translate("Form", "Submit"))
        self.browseButton.clicked.connect(self.browse_handler)
        self.submitButton.clicked.connect(self.submit_handler)
        if final_path != "":
            sys.exit()

    def browse_handler(self):
        self.open_dialog_box()

    def open_dialog_box(self):
        format = "Videos (*.mp4 *.avi *.mkv);"
        filename = QFileDialog.getOpenFileName(self, "Select Video", "", format)
        #filename = QFileDialog.getOpenFileName()
        path = filename[0]
        self.lineEdit.setText(path)

    def error_handler(self, error_message):
        error = QMessageBox()
        error.setIcon(QMessageBox.Warning)
        error.setWindowTitle("Error!")
        error.setText(error_message)
        error.exec()

    def submit_handler(self):
        if self.lineEdit.text() == '':
            self.error_handler("You Need to Choose a Video First!")
            return
        if Path(self.lineEdit.text()).suffix != '.mp4' and Path(self.lineEdit.text()).suffix != '.mkv':
            self.error_handler("The file you chose is not a video!")
            return
        path = self.lineEdit.text()
        global final_path
        final_path = path
        Form.close()

# This function returns the output layers of the network
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# this function draws a rectangle around the tortilla if found
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if label == "tortilla":
        return 1
    else:
        return 0


# This function checks if the tortilla is in the middle of the frame
def check_for_tortilla(image):
    img_x = image.shape[1]
    img_y = image.shape[0]

    whites_on_top = 0
    whites_on_bottom = 0

    # check for white pixels on top
    for i in range(0, img_x):
        if image[0][i] == 255:
            whites_on_top = 1
            break

    # check for white pixels on bottom
    for j in range(0, img_x):
        if image[img_y - 1][j] == 255:
            whites_on_bottom = 1
            break

    # if there are no white pixels on top and bottom or if there are both on top and bottom that means the tortilla is in the mddle
    if (whites_on_bottom == 1 and whites_on_top == 1) or (whites_on_bottom == 0 and whites_on_top == 0):
        return 1
    else:
        return 0

if __name__ == "__main__":

    final_path = ""

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)

    Form.show()
    app.exec_()
    files = glob.glob('results/*')
    for f in files:
        os.remove(f)

    if final_path == "":
        sys.exit()
    #we get the path to the video
    path = final_path
    # load the video
    cap = cv2.VideoCapture(path)

    count = 0
    classes = None

    # reads the classes from the txt
    with open("yolov3.txt", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # uses the configuration file and the pretrained weights to create the network
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # initialization of variables
    flag = 0
    circles_exist = 0
    frame_num = 0
    name = None
    image = None
    while (cap.isOpened()):
        count += 1
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            cv2.imwrite("temp.jpg", frame)
            Width = 720
            Height = 480
            scale = 0.00392
            img = cv2.imread("temp.jpg", 0)
            img = cv2.resize(img, (720, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
            # Display each frame
            cv2.imshow("frame", frame)
            # make the image black and white
            thresh, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            # copy the frame in order to present the object recognition in the frame2
            frame2 = frame.copy()

            val = 0
            # check the black and white image for white pixels
            if not (cv2.countNonZero(bw) == 0):
                # if there are white pixels we check if the tortilla is in the middle of the frame
                val = check_for_tortilla(bw)
                if val == 1:
                    # copy the frame where the circle recognition will be shown
                    frame3 = frame.copy()
                    gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

                    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    gray = cv2.medianBlur(gray, 5)

                    # Adaptive Guassian Threshold is to detect sharp edges in the Image.
                    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)
                    kernel = np.ones((3, 3), np.uint8)
                    gray = cv2.erode(gray, kernel, iterations=1)

                    # erosion
                    gray = cv2.dilate(gray, kernel, iterations=1)
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=30, param2=35, minRadius=0,
                                               maxRadius=0)
                    if circles is not None:
                        circles_exist = 1

                        # convert the (x, y) coordinates and radius of the circles to integers
                        circles = np.round(circles[0, :]).astype("int")

                        # loop over the (x, y) coordinates and radius of the circles
                        for (x, y, r) in circles:
                            # draw the circle in the output image, then draw a rectangle in the image
                            # corresponding to the center of the circle
                            cv2.circle(frame3, (x, y), r, (0, 255, 0), 4)

                        # Display and save the resulting frame
                        cv2.imwrite("results/frame" + str(count) + " circles.jpg", frame3)
                        cv2.imshow("circle detection", frame3)
                        if flag == 1:
                            name = "results/frame " + str(count) + ".jpg"
                            # font
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            # org
                            org = (50, 50)
                            # fontScale
                            fontScale = 1
                            # Blue color in BGR
                            color = (0, 255, 0)
                            # Line thickness of 2 px
                            thickness = 2
                            frame4 = frame.copy()
                            frame_num = count
                            # Using cv2.putText() method
                            image = cv2.putText(frame, 'good tortilla', org, font, fontScale, color, thickness,
                                                cv2.LINE_AA)
            else:
                flag = 0
                circles_exist = 0
                # this frame presents us if a tortilla is a good one or a bad one
                if not (name == None):
                    cv2.destroyWindow("tortilla detection")
                    cv2.destroyWindow("circle detection")
                    cv2.imwrite(name, image)
                    cv2.imshow("evaluation", image)
                    name = None
                    cv2.imwrite("results/original frame " + str(frame_num) + ".jpg", frame4)
            if val == 1 and flag == 0:

                blob = cv2.dnn.blobFromImage(frame2, scale, (416, 416), (0, 0, 0), True, crop=False)

                # we create the network
                net.setInput(blob)

                # we get the output layers
                outs = net.forward(get_output_layers(net))

                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.4

                # we check the image for an object that belongs in our classes
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])

                # the amount of objects found in the frame
                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                count_shapes = 0

                # for each item we draw our rectangle
                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    count_shapes += 1
                    # draw a rectangle if we find a tortilla
                    val = draw_prediction(frame2, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                                          round(y + h))
                    if val == 1 and count_shapes == 1:
                        flag = 1
                        cv2.imshow("tortilla detection", frame2)
                        # save the image with the tortilla recognition
                        name = "results/frame " + str(count) + ".jpg"
                        name2 = "results/frame " + str(count) + " tortilla recognition.jpg"
                        temp_frame = frame.copy()
                        cv2.imwrite(name2, frame2)
                        if circles_exist == 1:
                            # font
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            # org
                            org = (50, 50)
                            # fontScale
                            fontScale = 1
                            # Blue color in BGR
                            color = (0, 255, 0)
                            # Line thickness of 2 px
                            thickness = 2
                            frame4 = frame.copy()
                            frame_num = count
                            # Using cv2.putText() method
                            image = cv2.putText(frame, 'good tortilla', org, font, fontScale, color, thickness,
                                                cv2.LINE_AA)

                        else:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            # org
                            org = (50, 50)
                            # fontScale
                            fontScale = 1
                            # Blue color in BGR
                            color = (0, 0, 255)
                            # Line thickness of 2 px
                            thickness = 2
                            frame4 = frame.copy()
                            frame_num = count
                            # Using cv2.putText() method
                            image = cv2.putText(frame, 'bad tortilla', org, font, fontScale, color, thickness,
                                                cv2.LINE_AA)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            time.sleep(3)
            # cv2.waitKey(0)
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    os.remove("temp.jpg")