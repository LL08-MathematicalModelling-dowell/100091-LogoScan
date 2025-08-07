import json
import cv2
import matplotlib.pyplot as plt

def plot_bbox_from_coco(coco_json_path, image_file_name, image_path=None):
    """
    Plot bounding boxes from COCO format JSON for a specific image file name.
    
    Args:
        coco_json_path (str): Path to COCO format JSON file
        image_file_name (str): Name of the image file to plot bboxes for
        image_path (str, optional): Path to directory containing images. 
                                    If None, assumes image is in current directory.
    """
    # Load COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Find the image in the COCO data
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_file_name:
            image_info = img
            break
    
    if image_info is None:
        print(f"Image '{image_file_name}' not found in COCO JSON.")
        return
    
    # Get the image path
    img_path = image_path + '/' + image_file_name if image_path else image_file_name
    
    # Read the image
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Get annotations for this image
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_info['id']:
            annotations.append(ann)
    
    # Draw bounding boxes
    for ann in annotations:
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        
        # Convert to [x1, y1, x2, y2]
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add category label if available
        if 'category_id' in ann:
            category_id = ann['category_id']
            category_name = next(
                (cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id),
                str(category_id)
            )
            cv2.putText(image, category_name, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Bounding boxes for {image_file_name}")
    plt.show()

# Example usage:
plot_bbox_from_coco('modanet2018_instances_train.json', '0000174.jpg', 'process_dataset\\images\\train')