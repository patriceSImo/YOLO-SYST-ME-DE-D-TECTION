import cv2
import numpy as np

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

class YoloDetector:
    def __init__(self, config_path, weights_path, names_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.classes = []
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        result = [(class_ids[i], confidences[i], boxes[i]) for i in range(len(boxes)) if i in indexes]
        return result

class DetectionSystem:
    def __init__(self, image_source=None, config_path='yolov3.cfg', weights_path='yolov3.weights', names_path='coco.names'):
        self.loader = ImageLoader(image_source)
        self.detector = YoloDetector(config_path, weights_path, names_path)

    def process_image(self):
        image = self.loader.load_image()
        detections = self.detector.detect_objects(image)
        self.display_detections(image, detections)

    def process_video(self):
        capture = self.loader.load_video()
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            detections = self.detector.detect_objects(frame)
            self.display_detections(frame, detections)
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.loader.release_video()
        cv2.destroyAllWindows()

    def process_webcam(self):
        capture = self.loader.load_video()
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            detections = self.detector.detect_objects(frame)
            self.display_detections(frame, detections)
            cv2.imshow('Webcam Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.loader.release_video()
        cv2.destroyAllWindows()

    def display_detections(self, image, detections):
        for (class_id, confidence, box) in detections:
            x, y, w, h = box
            label = str(self.detector.classes[class_id])
            if label in ['mouse', 'book', 'money']:  # Ajoutez les étiquettes correspondant aux billets en euro
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == "__main__":
    system = DetectionSystem(
        config_path='yolov3.cfg',
        weights_path='yolov3.weights',
        names_path='coco.names'
    )
    system.process_webcam()  # Pour la détection en direct via la webcam
