from asyncio import wait_for
from .models import Person
from keras.models import load_model
import cv2
import face_recognition
import numpy as np
import requests
from datetime import datetime
import environ
import yagmail
import pywhatkit as kit
import pyautogui as pg
import time
from platform import system
from .PreprocessVideo import Preprocess

class CamPolice:
    def __init__(self, file):
        self.name = "0"
        self.video_file = file
        self.threshold = 50
        self.max_frames = 60
        self.area = self.get_location()

    
    def detect_violence(self, video):
        model = load_model('keras_model.h5', compile=False)
        is_fight = model.predict(np.asarray([video]))[0][0] * 100
        return is_fight
    

    def message(self, is_known, face_names, mail=True):
        if is_known and mail:
            self.mail_authorities([self.fetch_offender_details(face_name) for face_name in face_names])
        elif is_known:
            print(face_names)
            self.whatsapp_authorities([self.fetch_offender_details(face_name) for face_name in face_names])
            

    def find_faces(self, small_frames, known_face_encodings, known_face_names):
        is_known_face = False
        face_names = set()
        for rgb_small_frame in small_frames:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    self.name = known_face_names[best_match_index]
                    face_names.add(self.name)
                    is_known_face = True
        self.message(is_known_face, face_names, mail=False)
        

    def process_video(self):
        video_capture = cv2.VideoCapture(self.video_file)


        holland_image = face_recognition.load_image_file("faces_db/Tom_h.jpg")
        holland_face_encoding = face_recognition.face_encodings(holland_image)[0]

        tobey_image = face_recognition.load_image_file("faces_db/Tobey.png")
        tobey_face_encoding = face_recognition.face_encodings(tobey_image)[0]

        known_face_encodings = [
            holland_face_encoding,
            tobey_face_encoding
   
        ]
        known_face_names = [
            "2",
            "3"
        ]

        # Initialize some variables
        frames = []
        small_frames = []
        can_analyze = False
        f_count = 0

        while video_capture.isOpened():
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if ret:
                frame_for_v = cv2.resize(frame, (224,224), interpolation=cv2.INTER_AREA)
                frame_for_v = cv2.cvtColor(frame_for_v, cv2.COLOR_BGR2RGB)
                frame_for_v = np.reshape(frame_for_v, (224,224,3))
                frames.append(frame_for_v)
                f_count += 1
                if f_count >= self.max_frames:
                    frames = np.array(frames)
                    can_analyze = True
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                small_frames.append(rgb_small_frame)

                if can_analyze:
                    if self.detect_violence(Preprocess(self.video_file).prepare_data(frames)) >= self.threshold:
                        self.find_faces(small_frames, known_face_encodings, known_face_names)
                    can_analyze = False
                    frames = []
                    f_count = 0
                    small_frames = []
                   
                # Display the resulting image
                cv2.imshow('Video', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


    def fetch_offender_details(self, aadhar_id: str):
        persons = Person.objects.filter(aadhar_id=aadhar_id)
        return [(person.name, person.aadhar_id, person.address) for person in persons]


    def get_location(self):
        location = requests.get("https://ipinfo.io/").json()
        city = location["city"]
        state = location["region"]
        lattitude, longitude = location["loc"].split(",")
        return city, state, lattitude, longitude


    def mail_authorities(self, offender_details: list):
        message = f"A violent incident was observed in the area: {self.area} where the following offenders were involved: {offender_details}"
        env = environ.Env()
        environ.Env.read_env()
        # initiating connection with SMTP server
        yag = yagmail.SMTP(env("email"), env("psw"))
        yag.send(env("email"),f"{datetime.today()}: Violence Report from {self.area[0]}", message)
    

    def whatsapp_authorities(self, offender_details):
        env = environ.Env()
        environ.Env.read_env()
        message = f"A violent incident was observed in the area: {self.area} where the following offenders were involved: {offender_details}"
        kit.sendwhatmsg_instantly(env("phone"), message)
        pg.press("tab")
        time.sleep(1)
        pg.keyDown('shift')
        pg.press("tab")
        pg.keyUp('shift')
        time.sleep(1)
    
        """ At this point, the cursor prompt should be in the text box.  Pressing
        the ENTER key will cause the message to be sent
        """
        pg.press("enter")
        time.sleep(1)
        if system().lower() in ("windows", "linux"):
            pg.hotkey("ctrl", "w")
        elif system().lower() == "darwin":
            pg.hotkey("command", "w")
        else:
            raise Warning(f"{system().lower()} not supported!")
       
        
        
       

    