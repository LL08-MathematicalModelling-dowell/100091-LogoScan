import numpy as np
import pymongo
import json
from pathlib import Path



async def preload_features():
    features = None
    img_urls  = []
    print("Loading features and image URLs...")

# Load configuration
    config_json = json.loads(Path("DowellLogoScan/config.json").read_text())
    client = pymongo.MongoClient(host=config_json['mongo_path'])
    database = client['logoscan']
    feature_collection = database["features"]

    features_list = []
    img_urls_list = []

    for doc in feature_collection.find():
        feature = np.array(doc["feature"])
        image_id = doc.get("id")
        if image_id:
            image_url = f"https://liveuxstoryboard.com/image/{image_id}"
            features_list.append(feature)
            img_urls_list.append(image_url)

    features = np.array(features_list) if features_list else np.empty((0, 4096))
    img_urls = img_urls_list

    return features, img_urls
