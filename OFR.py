import cv2
import mediapipe as mp
import face_recognition
import numpy as np

mp_face_detection = mp.solutions.face_detection

known_face_encodings = []
known_face_names = []

known_images = [
    ("HR.jpg", "Hector Rosas"),
    ("ODG.jpg", "Oscar David"),
]

for image_path, name in known_images:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    process_this_frame = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) 
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            results = face_detection.process(rgb_small_frame)

            face_locations = []
            face_encodings = []
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = small_frame.shape
                    top = int(bbox.ymin * h)
                    left = int(bbox.xmin * w)
                    bottom = top + int(bbox.height * h)
                    right = left + int(bbox.width * w)
                    face_locations.append((top, right, bottom, left))

                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        cv2.imshow('Optimized Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
