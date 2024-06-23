import cv2

class ImageLoader:
    def __init__(self, source=None):
        self.source = source
        self.capture = None

    def load_image(self):
        if self.source:
            return cv2.imread(self.source)
        else:
            raise ValueError("No image source provided")

    def load_video(self):
        if self.source:
            self.capture = cv2.VideoCapture(self.source)
        else:
            self.capture = cv2.VideoCapture(0)  # 0 pour la webcam par défaut
        return self.capture

    def release_video(self):
        if self.capture:
            self.capture.release()
