import json
import os
import shutil
import random

def coco_to_yolo_using_existing_images(
    coco_json_path,
    image_dir,
    output_base_dir,
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)

    # Load COCO
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Map image_id -> image info
    image_info = {img['id']: img for img in coco['images']}
    category_id_to_class_id = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}

    # Group annotations by image ID
    annotations_by_image = {}
    for ann in coco['annotations']:
        image_id = ann['image_id']
        annotations_by_image.setdefault(image_id, []).append(ann)

    # Filter only existing image files
    existing_images = []
    for img_id, img in image_info.items():
        img_path = os.path.join(image_dir, img['file_name'])
        if os.path.exists(img_path):
            existing_images.append(img_id)
        else:
            print(f"Skipping missing image: {img['file_name']}")

    # Split
    random.shuffle(existing_images)
    split_index = int(len(existing_images) * train_ratio)
    train_ids = set(existing_images[:split_index])
    val_ids = set(existing_images[split_index:])

    # Create output folders
    for subfolder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(output_base_dir, subfolder), exist_ok=True)

    # Process images
    for img_id in existing_images:
        img = image_info[img_id]
        anns = annotations_by_image.get(img_id, [])

        file_name = img['file_name']
        file_stem = os.path.splitext(file_name)[0]
        src_img_path = os.path.join(image_dir, file_name)

        if img_id in train_ids:
            img_dst = os.path.join(output_base_dir, 'images/train', file_name)
            lbl_dst = os.path.join(output_base_dir, 'labels/train', file_stem + '.txt')
        else:
            img_dst = os.path.join(output_base_dir, 'images/val', file_name)
            lbl_dst = os.path.join(output_base_dir, 'labels/val', file_stem + '.txt')

        shutil.copy2(src_img_path, img_dst)

        with open(lbl_dst, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img['width']
                y_center = (y + h / 2) / img['height']
                w /= img['width']
                h /= img['height']
                class_id = category_id_to_class_id[ann['category_id']]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print("✅ Done. Only existing images were used and split into train/val sets.")

# Example usage:
coco_to_yolo_using_existing_images(
    coco_json_path="C:/Users/ahsan/Downloads/modanet-20250617T220826Z-1-001/modanet/modanet/modanet/annotations/modanet2018_instances_train.json",
    image_dir="C:\\Users\\ahsan\\Downloads\\modanet-20250617T220826Z-1-001\\modanet\\modanet\\coco\\images",
    output_base_dir='process_dataset',
    train_ratio=0.8
)