import pymongo
import json
from pathlib import Path

# Load configuration
config_json = json.loads(Path("config.json").read_text())
client = pymongo.MongoClient(host=config_json['mongo_path'], connectTimeoutMS=30000)
database = client['CCTV_Monitoring']