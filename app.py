from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import face_recognition
import os, sys
import numpy as np
import math
from external_libs.sort.sort import Sort
from external_libs.deep_sort.detection import Detection
from tools import generate_detections as gdet
from external_libs.deep_sort import nn_matching
from external_libs.deep_sort.tracker import Tracker
import configparser
from db import MongoDBHandler
from models.human import Human, HumanDic
import time
from datetime import datetime
from threading import Thread, Lock
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

app=Flask(__name__, static_url_path='/static')
config = configparser.ConfigParser()
config.read(os.path.abspath(os.path.join(".ini")))
app.config['DEBUG'] = True
app.config['MONGO_URI'] = config['PROD']['DB_URI']
app.config['MONGO_DB_NAME'] = config['PROD']['DB_NAME']
mongo_handler = MongoDBHandler(app.config)
socketio = SocketIO(app)
lock = Lock()
missing_minutes_threshold = 1
activity_time_period_start_str = "08:00"
activity_time_period_end_str = "19:00"

def get_timestamp(): 
    return time.time() * 1000

class KnownFace: 
    #status
    NORMAL = 0
    ABSENT = 1
    IN_THE_ROOM = 2 #out of activity time

    def __init__(self, name, staff=False):
        self.name = name
        self.status = self.NORMAL
        self.not_moving = False
        self.staff = staff
    
    def set_normal(self): 
        self.status = self.NORMAL
    
    def is_normal(self): 
        return self.status == self.NORMAL
    
    def set_absent(self): 
        self.status = self.ABSENT

    def is_absent(self): 
        return self.status == self.ABSENT
    
    def set_in_the_room(self): 
        self.status = self.IN_THE_ROOM
    
    def is_in_the_room(self): 
        return self.status == self.IN_THE_ROOM
    
    def set_moving(self, is_moving): 
        self.movinging = is_moving
    
    def is_moving(self): 
        return self.movinging 
    
    def is_staff(self): 
        return self.staff 
    
class FaceRecognition: 
    MIN_FACE_CONFIDENCE = 80
    HAND_GESTURE_TIME_THRESHOLD = 5000 # in seconds

    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    know_face_names = []
    know_face_human = {}
    human_locations = []
    process_current_frame = True
    use_deepsort = True
    activity_time_period_start = 0
    activity_time_period_end = 0

    hand_gestures = [] #storing hand gesture to trigger emergency call
    hand_gesture_count = 0 
    prev_hand_gesture_timestamp = 0

    # Load YOLO model
    net = cv2.dnn.readNet('models/yolov3/yolov3.weights', 'models/yolov3/yolov3.cfg')
    # Load classes
    classes = []
    tracker = Sort()
    current_track_id_list = []
    human_dic = HumanDic()

    #Deep SORT
    box_encoder = None
    deepsort_tracker = None

    def __init__(self):
        self.encode_faces()
        
        #YoloV3
        with open('models/yolov3/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f]
        #Deep SORT
        self.box_encoder = gdet.create_box_encoder('models/mars-small128.pb', batch_size=1)
        self.deepsort_tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", 0.7, None))
        self.update_activity_period_time()

    def process_hand_gesture(self, timestamp, gesture, frame): 
        # ["okay","peace","thumbs up","thumbs down","call me","stop","rock","live long","fist","smile"]
        # Opened Palm: 5 - stop, 7 - live long
        # Closed Palm: 4 - call me, 8 - fist
        #Opened Palm, Closed Palm, Opened Palm, Closed Palm
        if self.prev_hand_gesture_timestamp == 0 or (timestamp - self.prev_hand_gesture_timestamp) <= self.HAND_GESTURE_TIME_THRESHOLD:
            if (gesture == 5 or gesture == 7) and (self.hand_gesture_count%2 == 0): 
                #Opened Palm
                self.hand_gesture_count+=1
                self.prev_hand_gesture_timestamp = timestamp
            elif (gesture == 4 or gesture == 8) and (self.hand_gesture_count%2 == 1): 
                #Closed Palm
                self.hand_gesture_count+=1
                self.prev_hand_gesture_timestamp = timestamp

        radius = 20
        match self.hand_gesture_count:
            case 1:
                cv2.circle(frame, (radius+radius, radius+radius), radius, (0, 255, 0), thickness=-1) 
            case 2:
                cv2.circle(frame, (radius+radius, radius+radius), radius, (0, 255, 0), thickness=-1) 
                cv2.circle(frame, ((radius+radius)*2, radius+radius), radius, (0, 255, 0), thickness=-1)
            case 3:
                cv2.circle(frame, (radius+radius, radius+radius), radius, (0, 255, 0), thickness=-1) 
                cv2.circle(frame, ((radius+radius)*2, radius+radius), radius, (0, 255, 0), thickness=-1)
                cv2.circle(frame, ((radius+radius)*3, radius+radius), radius, (0, 255, 0), thickness=-1)
            case 4:
                cv2.circle(frame, (radius+radius, radius+radius), radius, (0, 0, 255), thickness=-1) 
                cv2.circle(frame, ((radius+radius)*2, radius+radius), radius, (0, 0, 255), thickness=-1)
                cv2.circle(frame, ((radius+radius)*3, radius+radius), radius, (0, 0, 255), thickness=-1)
                cv2.circle(frame, ((radius+radius)*4, radius+radius), radius, (0, 0, 255), thickness=-1)
                self.call_for_emergency()
            
    
    def call_for_emergency(self,):
        handle_call_for_emergency("CALL FOR EMERGENCY")
        self.hand_gesture_count = 0
        self.prev_hand_gesture_timestamp = 0

    def face_confidence(self, face_distance, face_match_threshold=0.6):
        range = (1.0-face_match_threshold)
        linear_val = (1.0 - face_distance)/(range*2.0)

        if face_distance > face_match_threshold: 
            return round(linear_val*100, 2)
        else: 
            value = (linear_val+((1.0-linear_val)*math.pow((linear_val-0.5)*2, 0.2)))*100
            return round(value, 2)
        
    def face_confidence_str(self, confidence): 
        return str(confidence)+'%'

    def encode_faces(self): 
        for image in os.listdir('faces'): 
            if not image.startswith('.'):
                face_image = face_recognition.load_image_file(f'faces/{image}')
                face_locations = face_recognition.face_encodings(face_image)
                if face_locations:
                    # Only proceed if at least one face is detected
                    face_encoding = face_recognition.face_encodings(face_image)[0]
                    self.known_face_encodings.append(face_encoding)
                    temp_name = image.split(".", 1)[0]
                    if '_' in temp_name: 
                        underscore_index = temp_name.index('_')
                        if underscore_index != -1 and temp_name[underscore_index + 1:] == "staff":
                            #staff
                            temp_name = temp_name[:underscore_index]
                            self.know_face_names.append(temp_name)
                            self.know_face_human[temp_name] = KnownFace(temp_name, True)
                    else: 
                        self.know_face_names.append(temp_name)
                        self.know_face_human[temp_name] = KnownFace(temp_name)
                else:
                    print(f"No faces found in {image}")

    def run_recognition(self): 
        video_capture = cv2.VideoCapture(0)
        '''
        for local video file use cv2.VideoCapture("video.mp4")
        for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
        for local webcam use cv2.VideoCapture(0)
        '''
        if not video_capture.isOpened():
            sys.exit('Video source not found')
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils
        model = load_model('models/mp_hand_gesture')

        while True:
            ret, frame = video_capture.read()
            current_timestamp = get_timestamp()
            alert_message = ""
            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            height, width, _ = small_frame.shape
            #to RGB
            rgb_small_frame = small_frame[:, :, ::-1]

            if self.prev_hand_gesture_timestamp > 0 and (current_timestamp - self.prev_hand_gesture_timestamp) > self.HAND_GESTURE_TIME_THRESHOLD: 
                #clear previous hand gesture
                self.prev_hand_gesture_timestamp = 0
                self.hand_gesture_count = 0

            #hand gesture detection
            result = hands.process(rgb_small_frame)
            # post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        #print(id, lm)
                        #lmx = int(lm.x * width)
                        #lmy = int(lm.y * height)
                        lmx = int(lm.x * height)
                        lmy = int(lm.y * width)
                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                    # Predict gesture
                    prediction = model.predict([landmarks], verbose=None)
                    # print(prediction)
                    classID = np.argmax(prediction)
                    self.process_hand_gesture(current_timestamp, classID, frame)

            #human and face detection
            
            if self.use_deepsort:
                #YOLOv3 + Deep SORT + face_recognition 
                if self.process_current_frame: 
                    alert_message = ""
                    # Prepare image for YOLO input
                    blob = cv2.dnn.blobFromImage(small_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                    self.net.setInput(blob)
                    # Get output layer names
                    output_layers = self.net.getUnconnectedOutLayersNames()

                    # Run forward pass to get detections
                    outs = self.net.forward(output_layers)
                    boxes = []
                    confidences = []
                    class_ids = []

                    # Process detections
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]

                            if confidence > 0.5 and class_id == 0:  # Assuming '0' is the class index for 'person'
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                # Append bounding box information to lists
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                                # Draw bounding box around the detected face
                                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Apply non-maximum suppression
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                    # Draw bounding boxes on the image
                    self.face_locations = []
                    self.face_names = []
                    self.human_locations = []
                    self.current_track_id_list = []
                    
                    for i in indices:
                        x, y, w, h = boxes[i]
                        temp_coord = (x, y, w, h)
                        #trackers = self.tracker.update(np.array([temp_coord]))
                        self.human_locations.append(temp_coord)
                    
                    if len(self.human_locations) == 0:
                        self.deepsort_tracker.predict()
                        self.deepsort_tracker.update([])
                    else: 
                        boxes = np.array(self.human_locations)
                        features = np.array(self.box_encoder(rgb_small_frame, boxes))
                        detection_objs = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
                        self.deepsort_tracker.predict()
                        self.deepsort_tracker.update(detection_objs)
                    
                    for track in self.deepsort_tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 5:
                            continue 
                        bbox = track.to_tlbr()
                        org_x, org_y, org_w, org_h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
                        track_id = track.track_id
                        # Extract the face region from the image
                        w = org_w if org_x>=0 else org_x+org_w
                        h = org_h if org_y>=0 else org_y+org_w
                        x = org_x if org_x>=0 else 0
                        y = org_y if org_y>=0 else 0
                        face_region = small_frame[y:y + h, x:x + w]
                        #rgb_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                        rgb_face_region = face_region[:, :, ::-1]
                        temp_face_locations = face_recognition.face_locations(rgb_face_region)
                        temp_face_encodings = face_recognition.face_encodings(rgb_face_region, temp_face_locations)
                        curr_name = ""
                        curr_confidence = 0
                        #Assume there is only one face in area of human 
                        #TODO: handle mutiple faces detected in one area 
                        #for index, face_encoding in enumerate(temp_face_encodings): 
                        if len(temp_face_encodings) > 0: 
                            face_encoding = temp_face_encodings[0]
                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                            name = 'Unknown'
                            confidence = 'Unknown'
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]: 
                                temp_confidence = self.face_confidence(face_distances[best_match_index])
                                if temp_confidence > self.MIN_FACE_CONFIDENCE: 
                                    curr_name = self.know_face_names[best_match_index]
                                    curr_confidence = temp_confidence
                                    name = self.know_face_names[best_match_index]
                                    confidence = self.face_confidence_str(temp_confidence)
                            self.face_names.append(f'{name}({confidence})')
                            self.face_locations.append((temp_face_locations[0][0]+y, temp_face_locations[0][1]+x, temp_face_locations[0][2]+y, temp_face_locations[0][3]+x))
                        if curr_name == "": 
                            self.human_dic.upsert_humans(track_id, True, curr_name, curr_confidence, current_timestamp, org_x, org_y, org_w, org_h)
                        else: 
                            self.human_dic.upsert_humans(track_id, False, curr_name, curr_confidence, current_timestamp, org_x, org_y, org_w, org_h)
                        self.current_track_id_list.append(track_id)

                    self.human_dic.delete_old_humans(self.current_track_id_list)  
                    not_move_name_list = self.human_dic.check_not_moving(current_timestamp)
                    for name, human in self.know_face_human.items():
                        if name in not_move_name_list: 
                            human.set_moving(False)
                        else: 
                            human.set_moving(True)
                    self.check_not_in_the_room(current_timestamp)
                    for name, human in self.know_face_human.items(): 
                        if human.is_staff(): 
                            if human.is_normal(): 
                                alert_message+="<span style='color:green;'>"+name+": staff </span><br>"
                            elif human.is_absent(): 
                                alert_message+="<span style='color:red;'>"+name+": absent </span><br>"
                        else: 
                            not_moving_str=""
                            if not human.is_moving(): 
                                not_moving_str = "(not moving)"
                            if human.is_normal(): 
                                if not_moving_str == "": 
                                    alert_message+="<span style='color:green;'>"+name+": normal</span><br>"
                                else: 
                                    alert_message+="<span style='color:red;'>"+name+": abnormal "+not_moving_str+"</span><br>"
                            elif human.is_absent(): 
                                alert_message+="<span style='color:red;'>"+name+": absent "+not_moving_str+"</span><br>"
                            elif human.is_in_the_room(): 
                                alert_message+="<span style='color:red;'>"+name+": in the room (abnormal) "+not_moving_str+"</span><br>"
                    handle_display_alert(alert_message)
                    mongo_handler.insert_detection(self.human_dic)

                #For debug. Printing recognition
                '''
                for(top, right, bottom, left), name in zip(self.face_locations, self.face_names): 
                    top*=4
                    right*=4
                    bottom*=4
                    left*=4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255),2)
                    cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255),-1)
                    cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255),1)
                '''
                
                #For debug. Printing YOLOv3 human location
                '''
                for(x, y, w, h) in self.human_locations:
                    cv2.rectangle(frame, (x*4, y*4), (x*4 + w*4, y*4 + h*4), (255, 0, 0), 6)
                '''
                
                for track in self.deepsort_tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 5:
                        continue 
                    bbox = track.to_tlbr()
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
                    track_id = track.track_id
                    x*=4
                    y*=4
                    w*=4
                    h*=4
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    human = self.human_dic.get_human_by_track_id(track_id)
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (0,0,255),-1)
                    cv2.putText(frame, human.get_name()+" "+human.get_confidence_str(), (x+6, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1) 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        '''
        #for standalone software
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1)==ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
        '''

    def change_inactivity_threshold(self, threshold): 
        self.human_dic.change_inactivity_threshold(threshold)
    
    def check_not_in_the_room(self, timestamp):
        absent_list = self.human_dic.get_not_detected(self.know_face_names)
        for name, human in self.know_face_human.items(): 
            if timestamp >= self.activity_time_period_start and timestamp <= self.activity_time_period_end:
                #should be here
                if name in absent_list: 
                    human.set_absent()
                else: 
                    human.set_normal()
            else: 
                #should not be here
                if name in absent_list:
                    human.set_normal()
                else: 
                    human.set_in_the_room()
    
    def update_activity_period_time(self): 
        today_date = datetime.now().date()
        temp_activity_time = datetime.strptime(activity_time_period_start_str, "%H:%M").time()
        self.activity_time_period_start = int(datetime.combine(today_date, temp_activity_time).timestamp() * 1000)
        temp_activity_time = datetime.strptime(activity_time_period_end_str, "%H:%M").time()
        self.activity_time_period_end = int(datetime.combine(today_date, temp_activity_time).timestamp() * 1000)         

def monitor_database():
    global missing_minutes_threshold
    know_face_names = []
    for image in os.listdir('faces'): 
        if not image.startswith('.'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            know_face_names.append(image.split(".", 1)[0])
    while True:
        with lock:
            names = mongo_handler.find_not_in_room(know_face_names, missing_minutes_threshold)
            alert_message = ', '.join(names)
            if alert_message != "":
                hours, remaining_minutes = divmod(missing_minutes_threshold, 60)
                time_str = ""
                if hours >= 1:
                    if remaining_minutes > 0: 
                        time_str = str(hours)+" hours "+str(remaining_minutes)+" minutes"
                    else: 
                        time_str = str(hours)+" hours "
                else: 
                    time_str = str(remaining_minutes)+" minutes"
                alert_message = "Not in room for "+time_str+": "+alert_message
            handle_display_not_in_room_alert(alert_message)
        socketio.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_capture')
def video_capture():
    return Response(fr.run_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('display_alert')
def handle_display_alert(alert_message):
    socketio.emit('display_alert', alert_message)

@socketio.on('display_not_in_room_alert')
def handle_display_not_in_room_alert(alert_message):
    socketio.emit('display_not_in_room_alert', alert_message)

@socketio.on('call_for_emergency')
def handle_call_for_emergency(alert_message):
    socketio.emit('call_for_emergency', alert_message)

@socketio.on('inactivity_duration_input')
def handle_input(input_value):
    fr.change_inactivity_threshold(int(input_value)*1000)

@socketio.on('not_in_room_input')
def handle_input(input_value):
    global missing_minutes_threshold
    with lock:
        missing_minutes_threshold = int(input_value)

@socketio.on('activity_start_time_input')
def handle_input(start_value, end_value):
    global activity_time_period_start_str
    global activity_time_period_end_str
    with lock:
        activity_time_period_start_str = str(start_value)
        activity_time_period_end_str = str(end_value)
        fr.update_activity_period_time()

if __name__=='__main__':
    fr = FaceRecognition()
    #database_thread = Thread(target=monitor_database)
    #database_thread.start()
    
    '''
    #for standalone software
    fr.run_recognition()
    '''

    #for web app
    socketio.run(app, debug=True,use_reloader=False, port=8000)