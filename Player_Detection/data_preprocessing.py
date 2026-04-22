import os
import json
import shutil

def format_soccernet_for_yolo(source_dir, output_dir):
    images_dir = os.path.join(output_dir, 'images', 'train')
    labels_dir = os.path.join(output_dir, 'labels', 'train')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    IMG_W, IMG_H = 1920, 1080

    for game_folder in os.listdir(source_dir):
        game_path = os.path.join(source_dir, game_folder)
        
        if not os.path.isdir(game_path) or not game_folder.startswith('SNGS'):
            continue

        img1_path = os.path.join(game_path, 'img1')
        json_path = os.path.join(game_path, 'Labels-GameState.json')

        if not os.path.exists(img1_path) or not os.path.exists(json_path):
            continue

        print(f"Processing {game_folder}...")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Map image_id to file_name
        id_to_filename = {str(img['image_id']): img['file_name'] for img in data['images']}

        # Group annotations by their target file_name
        image_annotations = {}
        for ann in data['annotations']:
            if 'bbox_image' not in ann:
                continue

            ann_img_id = str(ann['image_id'])
            
            if ann_img_id in id_to_filename:
                file_name = id_to_filename[ann_img_id]
                
                if file_name not in image_annotations:
                    image_annotations[file_name] = []
                
                image_annotations[file_name].append(ann)

        # Process the images and write labels
        all_images = sorted([f for f in os.listdir(img1_path) if f.endswith('.jpg')])
        
        if len(all_images) >= 3:
            # Grab the 1st, middle, and last frame
            middle_index = len(all_images) // 2
            images_to_process = [all_images[0], all_images[middle_index], all_images[-1]]
        else:
            images_to_process = all_images  # Fallback if the folder is weirdly empty

        # Loop through our selected 3 images
        for img_filename in images_to_process:
            
            img_id_base = img_filename.replace('.jpg', '')
            unique_name = f"{game_folder}_{img_id_base}"
            
            # Copy Image
            src_img = os.path.join(img1_path, img_filename)
            dst_img = os.path.join(images_dir, f"{unique_name}.jpg")
            shutil.copy(src_img, dst_img)

            # Write Label
            label_file_path = os.path.join(labels_dir, f"{unique_name}.txt")
            
            anns = image_annotations.get(img_filename, [])
            
            with open(label_file_path, 'w') as lf:
                for ann in anns:
                    attributes = ann.get('attributes') or {}
                    role = attributes.get('role', 'other')
                    team = attributes.get('team', 'none')
                    
                    if role == "player":
                        class_id = 0 if team == "left" else 1
                    elif role == "goalkeeper":
                        class_id = 2 if team == "left" else 3
                    else:
                        class_id = 4 

                    bbox = ann['bbox_image']
                    
                    x_c = bbox['x_center'] / IMG_W
                    y_c = bbox['y_center'] / IMG_H
                    w = bbox['w'] / IMG_W
                    h = bbox['h'] / IMG_H

                    lf.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    
    print(f"\n Done! Dataset is ready at: {output_dir}")


if __name__ == "__main__":
    
    # For training:
     
    SOURCE_DIR = 'SoccerNet/SN-GSR-2025/train'     
    OUTPUT_DIR = 'yolo_dataset_mini' 
    
    format_soccernet_for_yolo(SOURCE_DIR, OUTPUT_DIR)
    
    
    ## For validation
     
    SOURCE_DIR = 'SoccerNet/SN-GSR-2025/valid' 
    OUTPUT_DIR = 'yolo_dataset_valid_mini' 
    
    format_soccernet_for_yolo(SOURCE_DIR, OUTPUT_DIR)