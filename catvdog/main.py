from fastai.learner import load_learner
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
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
        self.inference_button = QPushButton("Run Inference", self)
        self.inference_button.clicked.connect(self.run_inference)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.image_label)
        layout.addWidget(self.inference_button)

    def run_inference(self):
        # You would typically load an image from your PySide6 app.
        # For simplicity, I'm using an example image here.
        image_path = './doge.jpeg'

        # Perform inference using the Fastai model
        img = cv2.imread(image_path)
        pred, _, _ = self.learn.predict(img)

        # Display the prediction result
        self.image_label.setText(f"Prediction: {pred}")

if __name__ == "__main__":
    app = QApplication([])
    window = MyMainWindow()
    window.show()
    app.exec()
