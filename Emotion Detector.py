import cv2
from deepface import DeepFace

# Load pre-trained models
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotionModel = DeepFace.build_model('Emotion')

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Loop through each frame in the video
while True:
    # Read the current frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the Haar cascade classifier
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Loop through each face detected in the frame
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y + h, x:x + w]

        # Preprocess the face image for emotion detection
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype('float') / 255.0
        face = face.reshape(1, 48, 48, 1)

        # Predict the emotion of the face using the DeepFace model
        result = emotionModel.predict(face)[0, :]
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_labels = dict(zip(emotions, result))
        dominant_emotion = max(emotion_labels, key=emotion_labels.get)

        # Draw a rectangle around the face and label it with the detected emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Wait for the 'q' key to be pressed to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
