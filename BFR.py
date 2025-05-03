import cv2
import mediapipe as mp
import face_recognition
import numpy as np

mp_face_detection = mp.solutions.face_detection

known_image = face_recognition.load_image_file("HR.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [known_encoding]
known_face_names = ["Hector Rosas"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_small_frame)

        if results.detections:
            face_locations = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = small_frame.shape
                top = int(bbox.ymin * h)
                left = int(bbox.xmin * w)
                bottom = top + int(bbox.height * h)
                right = left + int(bbox.width * w)
                face_locations.append((top, right, bottom, left))

            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(small_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Face Recognition', small_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
