import tensorflow as tf
import numpy as np
import cv2

class DigitPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = 119
        self.predicted_digit = ""
        self.roi_x_start = 320 - 75
        self.roi_x_end = 320 + 75
        self.roi_y_start = 240 - 75
        self.roi_y_end = 240 + 75
        self.image_width = 28
        self.image_height = 28


    def predict(self, img):
        img = np.expand_dims(img, axis=0)
        res = self.model.predict(img)
        index = np.argmax(res)
        return str(index)

    def on_threshold_change(self, x):
        self.threshold = x

    def process_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded_frame = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY_INV)
        roi = thresholded_frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        resized_roi = cv2.resize(roi, (self.image_width, self.image_height))
        return resized_roi, thresholded_frame

    def start_cv(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('background')
        cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('threshold', 'background', self.threshold, 255, self.on_threshold_change)
        background = np.zeros((480, 640, 3), np.uint8)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0

            roi_image, thresholded_frame = self.process_frame(frame)

            self.predicted_digit = self.predict(roi_image)

            display_frame = frame.copy()
            background[:] = display_frame[:]

            background[0:480, 0:80] = 0
            background[0:480, 560:640] = 0

            cv2.putText(background, self.predicted_digit, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            cv2.rectangle(background, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), (255, 255, 255), 3)

            cv2.imshow('background', background)

            cv2.imshow('Edges', thresholded_frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    model_path = '0-9CNN.keras'
    predictor = DigitPredictor(model_path)
    print('Loaded saved model.')
    print(predictor.model.summary())

    print("Starting CV...")
    predictor.start_cv()

main()
