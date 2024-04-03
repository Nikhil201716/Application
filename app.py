import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Real-time Face Detection and Counting Application")

    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Choose option", ["Upload Image", "Real-time Detection"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            perform_face_detection(image)
    elif option == "Real-time Detection":
        st.markdown("Click the button below to start real-time face detection:")
        if st.button("Start Detection"):
            real_time_face_detection()

def perform_face_detection(image):
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around faces and count the number of faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the image with bounding boxes and the number of faces detected
    st.image(image, channels="BGR", caption=f"Number of faces detected: {len(faces)}")

def real_time_face_detection():
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start video capture from the system's camera
    cap = cv2.VideoCapture(0)

    # Loop for real-time face detection
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes around faces and count the number of faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with bounding boxes and the number of faces detected
        st.image(frame, channels="BGR", caption=f"Number of faces detected: {len(faces)}")

        # Check for keyboard input to stop real-time detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
