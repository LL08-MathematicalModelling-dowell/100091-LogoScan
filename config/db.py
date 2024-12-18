import pymongo
import json
from pathlib import Path

# Load configuration
config_json = json.loads(Path("DowellLogoScan/config.json").read_text())
client = pymongo.MongoClient(host=config_json['mongo_path'])
database = client['logoscan']