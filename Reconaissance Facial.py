import cv2
import os
import face_recognition
import glob
import time

known_faces = []
known_names = []
known_faces_paths = []

# Load known faces and their encodings
registered_faces_path = 'registered/'
for name in os.listdir(registered_faces_path):
    images_mask = os.path.join(registered_faces_path, name, '*.jpg')
    images_paths = glob.glob(images_mask)
    known_faces_paths += images_paths
    known_names += [name] * len(images_paths)

def get_encodings(img_path):
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    return encoding[0] if encoding else None

known_faces = [get_encodings(img_path) for img_path in known_faces_paths]

vc = cv2.VideoCapture(0)  # Change the camera index as needed (0 for default camera)

# Set the desired frame rate (e.g., 10 frames per second)
desired_fps = 10
frame_delay = 0.2 / desired_fps  # Calculate the delay between frames

prev_frame_time = time.time()  # Initialize the previous frame time

while True:
    ret, frame = vc.read()
    if not ret:
        break

    # Calculate the time elapsed since the previous frame
    current_time = time.time()
    elapsed_time = current_time - prev_frame_time

    # Process the frame only if enough time has passed to achieve the desired frame rate
    if elapsed_time >= frame_delay:
        prev_frame_time = current_time  # Update the previous frame time

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(frame_rgb)

    for face in faces:
        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        encoding = face_recognition.face_encodings(frame_rgb, [face])[0]

        results = face_recognition.compare_faces(known_faces, encoding)
        if any(results):
            name = known_names[results.index(True)]
        else:
            name = 'unknown'
        
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow('Facial Recognition', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()