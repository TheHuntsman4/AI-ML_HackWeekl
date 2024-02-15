from fastai.learner import load_learner
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PySide6.QtGui import QImage, QPixmap, QImageReader, QImageIOHandler
import cv2
from pathlib import Path

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the Fastai model
        self.learn = load_learner('./model.pkl')

        # Set up GUI elements
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAcceptDrops(True) 
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 2px dashed #aaa;")

        self.inference_button = QPushButton("Run Inference", self)
        self.inference_button.clicked.connect(self.run_inference)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.image_label)
        layout.addWidget(self.inference_button)

        # Connect drag and drop events
        self.image_label.dragEnterEvent = self.dragEnterEvent
        self.image_label.dropEvent = self.dropEvent

    def run_inference(self):
        if hasattr(self, 'current_image_path'):
            image_path = self.current_image_path

            # Perform inference using the Fastai model
            img = cv2.imread(image_path)
            pred, _, _ = self.learn.predict(img)

            # Display the prediction result
            self.image_label.setText(f"Prediction: {pred}")

    def dragEnterEvent(self, event):
        mime_data = event.mimeData()

        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            event.acceptProposedAction()

    def dropEvent(self, event):
        mime_data = event.mimeData()

        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            file_path = mime_data.urls()[0].toLocalFile()
            self.load_image(file_path)

    def load_image(self, file_path):
        self.current_image_path = file_path
        image = QImage(file_path)
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")

if __name__ == "__main__":
    app = QApplication([])
    window = MyMainWindow()
    window.show()
    app.exec()
