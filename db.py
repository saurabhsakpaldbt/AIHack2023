from pymongo import MongoClient
from datetime import datetime, timedelta
from models.human import Human, HumanDic

class MongoDBHandler:
    client = None

    @classmethod
    def init_client(cls, config):
        if not cls.client and config:
            cls.client = MongoClient(config.get('MONGO_URI'))
    
    @classmethod
    def close_client(cls):
        if cls.client:
            cls.client.close()

    def __init__(self, config):
        self.init_client(config)
        self.db = MongoDBHandler.client[config.get('MONGO_DB_NAME')]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def insert_detection(self, human_dic):
        collection = self.db["detections"]

        detections = [
            {
                "track_id": human.track_id,
                "unknown": human.unknown,
                "name": human.name,
                "confidence": human.confidence, 
                "x": human.x, 
                "y": human.y, 
                "w": human.w, 
                "h": human.h, 
                "status": human.status
            }
            for key, human in human_dic.humans.items()
        ]

        detection_data = {
            "timestamp": datetime.utcnow(),
            "detections": detections
        }

        collection.insert_one(detection_data)

    def find_not_in_room(self, name_list, threshold):
        collection = self.db["detections"]
        specific_time = datetime.utcnow() - timedelta(minutes=threshold)
        query = {
            'timestamp': {'$gt': specific_time},
            'detections.name': {'$in': name_list},
        }
        results = collection.find(query)
        names_set = set(name_list)
        for result in results:
            names_in_room = [detection['name'] for detection in result.get('detections', [])]
            names_set -= set(names_in_room)
        return list(names_set)