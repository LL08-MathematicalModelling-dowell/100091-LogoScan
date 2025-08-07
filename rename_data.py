import os

def rename_yolo_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir):
    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Get all image files (assuming extensions like .jpg, .png, etc.)
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    # Rename and copy images and labels sequentially starting from 1
    for i, image_file in enumerate(image_files, start=1):
        # Original paths
        original_image_path = os.path.join(images_dir, image_file)
        original_label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')
        
        # New paths
        new_image_name = f"{i}.{image_file.split('.')[-1]}"  # e.g., 1.jpg, 2.jpg, etc.
        new_label_name = f"{i}.txt"  # e.g., 1.txt, 2.txt, etc.
        
        new_image_path = os.path.join(output_images_dir, new_image_name)
        new_label_path = os.path.join(output_labels_dir, new_label_name)
        
        # Rename (move) the files
        os.rename(original_image_path, new_image_path)
        if os.path.exists(original_label_path):
            os.rename(original_label_path, new_label_path)
        else:
            print(f"Warning: Label file not found for {image_file}")

    print(f"Renaming complete. Files are saved in {output_images_dir} and {output_labels_dir}")

# Example usage:
images_dir = "four_dataset/images/val"       # Folder containing original images
labels_dir = "four_dataset/labels/val"       # Folder containing original labels
output_images_dir = "cl_four_dataset/images/val"  # Folder to save renamed images (can be same as images_dir)
output_labels_dir = "cl_four_dataset/labels/val"  # Folder to save renamed labels (can be same as labels_dir)

rename_yolo_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir)