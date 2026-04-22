import json
import os
from ultralytics import YOLO

DEFAULT_MODEL_PATH = 'runs/detect/soccernet_m5/weights/best.pt'

class Person2Detection:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model = YOLO(model_path)
        
        self.class_names = {
            0: "player_team_a", 
            1: "player_team_b", 
            2: "goalkeeper_team_a", 
            3: "goalkeeper_team_b", 
            4: "other"
        }

    def process_sequence(self, image_folder, output_json):
        sequence_results = {}

        # Sorted list of images
        images = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])

        for img_name in images:
            img_path = os.path.join(image_folder, img_name)
            results = self.model(img_path, conf=0.3, verbose=False)[0]
            
            frame_detections = []
            for box in results.boxes:
                # box coordinates in pixels [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = coords

                # bottom-middle of the box
                foot_x = (x1 + x2) / 2
                foot_y = y2 

                frame_detections.append({
                    "class_id": int(box.cls[0]),
                    "label": self.class_names.get(int(box.cls[0]), "unknown"),
                    "confidence": float(box.conf[0]),
                    "bbox_image": [round(x, 2) for x in coords],
                    "foot_point": [round(foot_x, 2), round(foot_y, 2)] # For Person 3
                })
            
            sequence_results[img_name] = frame_detections

        with open(output_json, 'w') as f:
            json.dump(sequence_results, f, indent=4)
        print(f" Data saved in {output_json}")

if __name__ == "__main__":
    detector = Person2Detection()
    
    mapping_folder = 'mapping_data' 
    if not os.path.exists(mapping_folder):
        os.makedirs(mapping_folder)
    
    # Process one game for testing purposes
    game_folder = 'SoccerNet/SN-GSR-2025/train/SNGS-060/img1'
    detector.process_sequence(game_folder, 'SNGS-001_detections.json')
    
    """ 
    import json

    path = "SoccerNet/SN-GSR-2025/train/sequences_info.json"

    with open(path, 'r') as f:
        data = json.load(f)
    file_names = [item['name'] for item in data['train']]
    
    base_path = 'SoccerNet/SN-GSR-2025/train'

    for file_name in file_names:
        folder_path = os.path.join(base_path, file_name, 'img1')
        
        if os.path.exists(folder_path):
            print(f"Processing {file_name}...")
            output_path = os.path.join(mapping_folder, f"{file_name}_detections.json")
            detector.process_sequence(folder_path, output_path)
        else:
            print(f"Warning: Folder not found for {file_name}") 
    """
        