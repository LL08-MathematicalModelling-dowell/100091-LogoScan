import numpy as np
from .db import database



async def preload_features():
    features = None
    print("Loading features...")

# Load configuration
    feature_collection = database["features"]

    features_list = []

    for doc in feature_collection.find():
        feature = np.array(doc["feature"])
        image_id = doc.get("id")
        if image_id:
            features_list.append(feature)


    features = np.array(features_list) if features_list else np.empty((0, 4096))

    return features
