import cv2
import matplotlib.pyplot as plt

def plot_yolo_bboxes(image_path, label_path, class_names=None):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Read YOLO annotations
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])

        # Convert from relative coordinates to absolute coordinates
        x_center = x_center * w
        y_center = y_center * h
        bbox_width = bbox_width * w
        bbox_height = bbox_height * h

        # Calculate coordinates
        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_height / 2)

        # Draw rectangle
        color = (255, 0, 0)  # Red in RGB
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = class_names[class_id] if class_names else f"Class {class_id}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage
plot_yolo_bboxes('process_dataset\\images\\train\\0000174.jpg', 
                 'process_dataset\\labels\\train\\0000174.txt', 
                 class_names=['bag', 'belt', 'boots', 'footwear', 'outer', 
                             'dress', 'sunglasses', 'pants', 'top', 'shorts', 
                             'skirt', 'headwear', 'scarf & tie'])