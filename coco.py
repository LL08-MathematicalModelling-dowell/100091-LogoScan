import os
import json

# Paths
images_dir = "C:\\Users\\ahsan\\Downloads\\modanet-20250617T220826Z-1-001\\modanet\\modanet\\coco\\images"  # folder with actual images
json_path = 'modanet2018_instances_train.json'  # original COCO JSON
output_json = 'modanet_all_data.json'  # output file for all data

# Load JSON
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Get filenames in images folder
existing_files = set(os.listdir(images_dir))

# Keep only images that exist in the folder
filtered_images = [img for img in coco_data['images'] if img['file_name'] in existing_files]
filtered_ids = [img['id'] for img in filtered_images]

# Filter annotations to only include those for existing images
filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in filtered_ids]

# Create combined JSON with all data
combined_json = {
    'images': filtered_images,
    'annotations': filtered_annotations,
    'categories': coco_data['categories']
}

# Save the combined JSON
with open(output_json, 'w') as f:
    json.dump(combined_json, f)

print(f"Filtered {len(filtered_images)} images that exist in '{images_dir}'")
print(f"Saved to: {output_json}")