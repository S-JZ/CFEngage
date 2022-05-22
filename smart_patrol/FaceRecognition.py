from .models import Person

class CamPolice:
    def detect_faces():
        import cv2
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX 
        font_scale = 1
        font_color = (0,0,0)
        stroke = 2
        cam = cv2.VideoCapture(0)
        while True:
            success, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            cv2.putText(frame, "Number of Faces: " + str(len(faces)), (50, 50), font, font_scale, font_color, stroke, cv2.LINE_AA)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Face Detection", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()

    def fetch_offender_details(face_encoding: str):
        persons = Person.objects.filter(face_encoding=face_encoding)
        return [(person.name, person.aadhar_id, person.address) for person in persons]
    
    def mail_authorities(offender_details: list):
        pass

    