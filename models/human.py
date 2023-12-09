import numpy as np

class Human: 
    DISTANCE_THRESHOLD = 7
    
    #status
    NORMAL = "n"
    INACTIVE = "I"

    def __init__(self, track_id, unknown, name, confidence,timestamp, x, y, w, h):
        self.track_id = track_id 
        self.unknown = unknown
        self.name = name
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.prev_position = (timestamp, self.get_centroid())
        self.status = self.NORMAL

    def get_name(self):
        if self.name == "": 
            return "Unknown"
        else: 
            return self.name
        
    def get_confidence_str(self): 
        return str(self.confidence)+'%'
    
    def get_centroid(self):
        return ((self.x+self.w)/2, (self.y+self.h)/2)
    
    def update_position(self, timestamp): 
        distances = [np.linalg.norm(np.array(self.get_centroid()) - np.array(self.prev_position[1]))]
        #print(distances)
        if distances[0] > self.DISTANCE_THRESHOLD: 
            #moving. Update position
            self.prev_position = (timestamp, self.get_centroid())

    def is_not_move(self, timestamp, threshold):
        not_move = timestamp - self.prev_position[0] > threshold 
        if not_move: 
            self.status = self.INACTIVE
        else: 
            self.status = self.NORMAL
        return not_move 

    def __str__(self):
        return "Track ID: {}, Unknown: {}, Name: {}, Confidence: {}, x: {}, y: {}, w: {}, h: {}".format(self.track_id, self.unknown, self.name, self.confidence, self.x, self.y, self.w, self.h)
        #return f"Track ID: {self.track_id}, Unknown: {self.unknown}, Name: {self.name}, Confidence: {self.confidence}"
    
class HumanDic: 
    inactivity_threshold = 10000 #10s

    def __init__(self):
        self.humans = {}

    def add_human(self, human):
        self.humans[human.track_id] = human
    
    def delete_human(self, track_id):
        if track_id in self.humans:
            del self.humans[track_id]
    
    def upsert_humans(self, track_id, new_unknown, new_name, new_confidence, timestamp, x, y, w, h):
        if track_id in self.humans:
            temp_human = self.humans[track_id]
            if temp_human.unknown:  
                #Assume face recognised 100% correct, name will not be updated.  
                #TODO: handle name changed
                temp_human.unknown = new_unknown
                temp_human.name = new_name
            #update confidence for recording
            temp_human.confidence = new_confidence
            temp_human.x = x
            temp_human.y = y
            temp_human.w = w
            temp_human.h = h
            temp_human.update_position(timestamp)
        else: 
            self.add_human(Human(track_id, new_unknown, new_name, new_confidence, timestamp, x, y, w, h))
    
    def delete_old_humans(self, current_track_ids): 
        old_track_ids = set(self.humans.keys()) - set(current_track_ids)
        for track_id in old_track_ids:
            del self.humans[track_id]

    def get_human_by_track_id(self, track_id):
        return self.humans.get(track_id)
    
    '''
    def check_not_moving(self, timestamp):
        alert_str = ""
        for key, human in self.humans.items(): 
            if human.is_not_move(timestamp, self.inactivity_threshold): 
                name = human.name
                if name == "": 
                    name = "Unknown"
                alert_str += name+" is not moving<br>"
        return alert_str
    '''
    
    def check_not_moving(self, timestamp):
        name_list = set()
        for key, human in self.humans.items(): 
            if human.is_not_move(timestamp, self.inactivity_threshold): 
                name = human.name
                if name == "": 
                    name = "Unknown"
                name_list.add(name)
        return name_list

    def change_inactivity_threshold(self, threshold): 
        self.inactivity_threshold = threshold
    
    def get_not_detected(self, name_list): 
        human_names_set = set(human.name for human in self.humans.values())
        return [name for name in name_list if name not in human_names_set]

    def __str__(self):
        h_list = "Human List:\n' + '------------\n"
        for key, value in self.humans.items():
            h_list += str(value)
        return h_list