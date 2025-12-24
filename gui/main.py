import sys
import os
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

from ultralytics import YOLO


class YOLOGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Object Detection GUI")
        self.setGeometry(100, 100, 1200, 650)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "model", "best.pt")

        if not os.path.exists(self.model_path):
            QMessageBox.critical(self, "Error", "best.pt not found in model folder")
            sys.exit(1)

        self.model = YOLO(self.model_path)

        self.image = None
        self.result_image = None

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.source_type = None

        self.init_ui()

    def init_ui(self):
        self.lbl_original_title = QLabel("Original Image", self)
        self.lbl_original_title.setGeometry(220, 20, 300, 30)
        self.lbl_original_title.setAlignment(Qt.AlignCenter)

        self.lbl_result_title = QLabel("Tagged Image", self)
        self.lbl_result_title.setGeometry(680, 20, 300, 30)
        self.lbl_result_title.setAlignment(Qt.AlignCenter)

        self.lbl_original = QLabel("No Image Loaded", self)
        self.lbl_original.setGeometry(100, 60, 450, 350)
        self.lbl_original.setStyleSheet("border: 2px solid gray;")
        self.lbl_original.setAlignment(Qt.AlignCenter)

        self.lbl_result = QLabel("No Detection Yet", self)
        self.lbl_result.setGeometry(650, 60, 450, 350)
        self.lbl_result.setStyleSheet("border: 2px solid gray;")
        self.lbl_result.setAlignment(Qt.AlignCenter)

        self.btn_select_image = QPushButton("Select Image", self)
        self.btn_select_image.setGeometry(100, 440, 200, 40)
        self.btn_select_image.clicked.connect(self.select_image)

        self.btn_test_image = QPushButton("Test Image", self)
        self.btn_test_image.setGeometry(320, 440, 200, 40)
        self.btn_test_image.clicked.connect(self.test_image)

        self.btn_save_image = QPushButton("Save Image", self)
        self.btn_save_image.setGeometry(650, 440, 200, 40)
        self.btn_save_image.clicked.connect(self.save_image)

        self.btn_test_video = QPushButton("Test Video", self)
        self.btn_test_video.setGeometry(870, 440, 200, 40)
        self.btn_test_video.clicked.connect(self.toggle_video)

        self.btn_test_camera = QPushButton("Test Camera", self)
        self.btn_test_camera.setGeometry(480, 500, 240, 40)
        self.btn_test_camera.clicked.connect(self.toggle_camera)

        self.lbl_info = QLabel("", self)
        self.lbl_info.setGeometry(100, 560, 1000, 40)
        self.lbl_info.setAlignment(Qt.AlignCenter)

    # =========================
    def show_image(self, label, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(), Qt.KeepAspectRatio
        )
        label.setPixmap(pix)

    # =========================
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.image = cv2.imread(path)
            self.show_image(self.lbl_original, self.image)
            self.lbl_info.setText("Image loaded")

    def test_image(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Select image first")
            return

        results = self.model(self.image)[0]
        self.result_image = results.plot()
        self.show_image(self.lbl_result, self.result_image)

        self.update_detection_info(results)

    def save_image(self):
        if self.result_image is None:
            QMessageBox.warning(self, "Warning", "No result to save")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self.result_image)
            QMessageBox.information(self, "Saved", "Image saved successfully")

    # =========================
    def toggle_video(self):
        if self.cap is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Video (*.mp4 *.avi)"
            )
            if not path:
                return
            self.cap = cv2.VideoCapture(path)
            self.source_type = "video"
            self.timer.start(30)
            self.btn_test_video.setText("Stop Video")
        else:
            self.stop_stream()

    def toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.source_type = "camera"
            self.timer.start(30)
            self.btn_test_camera.setText("Stop Camera")
        else:
            self.stop_stream()

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_stream()
            return

        results = self.model(frame)[0]
        frame = results.plot()
        self.show_image(self.lbl_result, frame)
        self.update_detection_info(results)

    def update_detection_info(self, results):
        counts = {}
        names = results.names
        for c in results.boxes.cls.tolist():
            name = names[int(c)]
            counts[name] = counts.get(name, 0) + 1

        if counts:
            info = " | ".join([f"{k}: {v}" for k, v in counts.items()])
            self.lbl_info.setText(f"Detected -> {info}")
        else:
            self.lbl_info.setText("Detected -> None")

    def stop_stream(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_test_video.setText("Test Video")
        self.btn_test_camera.setText("Test Camera")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOGui()
    window.show()
    sys.exit(app.exec_())
