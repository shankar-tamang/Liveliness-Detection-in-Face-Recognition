import dlib
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the HOG face detector
detector = dlib.get_frontal_face_detector()

# Load your liveness detection model
try:
    model = load_model('liveliness_detection_model_resnet.h5')
    # model = load_model('liveliness_detection_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to preprocess input for the liveness detection model
def preprocess_input(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict liveness
def predict_liveness(image):
    preprocessed_image = preprocess_input(image)
    prediction = model.predict(preprocessed_image)
    print(prediction)
    return "Live" if prediction[0][0] > 0.44 else "Spoof"

# Main function for image and video processing
def main():
    input_type = int(input("Enter 1 for image or 2 for video: "))

    if input_type == 1:  # Image
        image_path = input("Enter image path: ")
        image = cv2.imread(image_path)
        if image is None:
            print("Error loading image.")
            return

        # Detect faces using HOG
        faces = detector(image, 1)

        # Process each detected face
        for face in faces:
            x1, y1, x2, y2 = max(0, face.left()), max(0, face.top()), min(image.shape[1], face.right()), min(image.shape[0], face.bottom())
            face_roi = image[y1:y2, x1:x2]

            # Predict liveness
            label = predict_liveness(face_roi)

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the result
        cv2.imwrite("liveness_detection_result.jpg", image)
        print("Result saved as 'liveness_detection_result.jpg'.")

    elif input_type == 2:  # Video
        video_capture = cv2.VideoCapture(0)  # Use 0 for default camera
        if not video_capture.isOpened():
            print("Error opening video capture.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error reading frame.")
                break

            # Detect faces using HOG
            faces = detector(frame, 1)

            # Process each detected face
            for face in faces:
                x1, y1, x2, y2 = max(0, face.left()), max(0, face.top()), min(frame.shape[1], face.right()), min(frame.shape[0], face.bottom())
                face_roi = frame[y1:y2, x1:x2]

                # Predict liveness
                label = predict_liveness(face_roi)

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the result
            cv2.imshow("Liveness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    else:
        print("Invalid input type.")

if __name__ == "__main__":
    main()
