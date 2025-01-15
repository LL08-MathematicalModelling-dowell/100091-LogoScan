import pymongo
import os
MONGO_PATH = os.environ.get("MONGO_PATH")

client = pymongo.MongoClient(host = MONGO_PATH )
database = client['ocr']