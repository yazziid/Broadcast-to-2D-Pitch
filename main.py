import cv2
import glob
import json

from ultralytics import YOLO
from pathlib import Path

from Player_Detection.detection2mapping import Person2Detection
from metrics.player_detection_metrics import evaluate_game
from mapping_2d.mapping import auto_map_detections
from metrics.mapping_metrics import evaluate_mapping

MODEL_WEIGHTS = {
    "YOLO": "runs/detect/soccernet_m5/weights/best.pt",
    "DETR": "runs/detect/soccernet_rtdetr/weights/best.pt"
}

BASE_DIR = Path("Test Values")

def get_paths(game_id, model_choice, base_directory = BASE_DIR):
    """This function generates all relevant paths based on game ID and model choice."""
    
    game_folder = base_directory / game_id
    input_images_path = game_folder / "img1"
    
    
    # The output structure looks as follows: Test Values / <Game> / <Model> / Player_Detection
    output_root = game_folder / model_choice / "Player_Detection"
    image_output_folder = output_root / "images"
    json_output_path = output_root / f"{game_id}_detections.json"
    mapped_output_folder = game_folder / model_choice / "mapped_data"
    mapped_json_output_path = game_folder / model_choice / f"{game_id}_pitch_positions_auto.json"
    ground_truth_json_path = game_folder / "Labels-GameState.json"
    
    return {
        "input": input_images_path,
        "output_root": output_root,
        "images": image_output_folder,
        "json": json_output_path,
        "mapped_data": mapped_output_folder,
        "mapped_json": mapped_json_output_path,
        "weights": MODEL_WEIGHTS.get(model_choice),
        "ground_true_detections": ground_truth_json_path
    }

def frames2video(paths, game_id, model_choice):
    image_folder = paths["images"]
    output_images = sorted(glob.glob(str(image_folder / "*.jpg")))
    
    if not output_images:
        print(f"Error: No images found in {image_folder}")
        return
    
    frame = cv2.imread(output_images[0])
    height, width, _ = frame.shape
    
    video_name = f"{game_id}_{model_choice}_results.mp4"
    save_path = image_folder.parent / video_name
    
    video = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

    for image_path in output_images:
        img = cv2.imread(image_path)
        if img is not None:
            video.write(img)

    video.release()
    print(f"Video saved to: {save_path}")
    
# Runs the fine tuned model on 
def detect_players(paths):
    model = YOLO(paths["weights"])
    
    output_dir = paths["images"]
    output_dir.mkdir(parents=True, exist_ok=True)
 
    results = model.predict(source=str(paths["input"]), conf=0.25, imgsz=640, stream=True )
    
    for result in results:
        file_name = Path(result.path).name
        save_dest = output_dir / file_name
        result.save(filename=str(save_dest))
        
    print(f"Detection frames saved to: {output_dir}")

# Run the detection to mapping function. Input the model choice and the game folder we wish to map.
def run_detection2mapping(paths, game_id):    
    detector = Person2Detection(paths["weights"])
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    
    print(f"Generating detection data for {game_id}...")
    detector.process_sequence(str(paths["input"]), str(paths["json"]))
    
    print(f"Data saved to: {paths['json']}")
    return paths["json"]      


def run_2d_mapping(paths, game_id):
    detections_path = paths["json"]
    mapped_output_dir = paths["mapped_data"]
    mapped_output_dir.mkdir(parents=True, exist_ok=True)

    if not detections_path.exists():
        raise FileNotFoundError(
            f"Detection JSON not found for {game_id}: {detections_path}"
        )

    print(f"Running 2d mapping for {game_id}...")
    results = auto_map_detections(
        detections_path=detections_path,
        output_path=paths["mapped_json"],
        visualization_dir=mapped_output_dir,
        alpha=0.8,
        max_frames=None,
        images_dir=paths["input"],
    )
    print(f"Mapped {len(results['frames'])} frames.")
    print(f"Mapped JSON saved to: {paths['mapped_json']}")
    print(f"Mapped frames saved to: {mapped_output_dir}")
    return paths["mapped_json"]

def combine_side_by_side(paths, game_id, model_choice):
    """Combines YOLO detection frames and 2D mapped pitch frames side-by-side."""
    left_dir = paths["images"]           # YOLO Detections
    right_dir = paths["mapped_data"]     # 2D Pitch Maps
    
    # save in Test Values/<GAME_ID>/<MODEL>/Player_Detection_and_mapping
    output_dir = left_dir.parent.parent / "Player_Detection_and_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    left_images = sorted(glob.glob(str(left_dir / "*.jpg")))
    
    if not left_images:
        print(f"Error: No images found in {left_dir}")
        return
        
    print(f"Combining side-by-side images for {game_id}...")
    
    match_count = 0
    for left_path in left_images:
        left_name = Path(left_path).name
        stem = Path(left_path).stem  
        
        right_path = right_dir / f"{stem}_pitch.png"
        
        if not right_path.exists():
            continue  
            
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(str(right_path))
        
        if img_left is None or img_right is None:
            continue
            
        h_left = img_left.shape[0]
        h_right, w_right = img_right.shape[:2]
        
        aspect_ratio = w_right / h_right
        new_w_right = int(h_left * aspect_ratio)
        
        img_right_resized = cv2.resize(img_right, (new_w_right, h_left), interpolation=cv2.INTER_AREA)
        
        combined = cv2.hconcat([img_left, img_right_resized])
        
        save_dest = output_dir / left_name
        cv2.imwrite(str(save_dest), combined)
        match_count += 1
        
    print(f"Successfully combined {match_count} frames.")
    print(f"Saved to: {output_dir}")


def side_by_side2video(paths, game_id, model_choice, fps = 25):
    """Converts the side-by-side combined frames into an MP4 video."""
    side_by_side_folder = paths["images"].parent.parent / "Player_Detection_and_mapping"
    
    output_images = sorted(glob.glob(str(side_by_side_folder / "*.jpg")))
    
    if not output_images:
        print(f"Error: No images found in {side_by_side_folder}. Did you run combine_side_by_side first?")
        return
    
    frame = cv2.imread(output_images[0])
    height, width, _ = frame.shape
    
    # Save the video directly into the same Player_Detection_and_mapping folder
    video_name = f"{game_id}_{model_choice}_side_by_side.mp4"
    save_path = side_by_side_folder / video_name
    
    print(f"Encoding side-by-side video for {game_id} at {fps} FPS...")
    
    video = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_path in output_images:
        img = cv2.imread(image_path)
        if img is not None:
            video.write(img)

    video.release()
    print(f"Side-by-side video successfully saved to: {save_path}")


if __name__ == "__main__": 
    GAME_ID = "SNGS-128"
    MODEL_CHOICE = "DETR"  # Options: "YOLO" or "DETR"

    # Generate all dynamic paths
    paths = get_paths(GAME_ID, MODEL_CHOICE)

    # Player Detection
    detect_players(paths)

    # Convert saved frames to Video
    # frames2video(paths, GAME_ID, MODEL_CHOICE)

    # mapping to JSON
    detection_json = run_detection2mapping(paths, GAME_ID)
   
    # Metrics
    evaluate_game(paths["ground_true_detections"], paths["json"], MODEL_CHOICE)
    
    # 2-D mapping
    mapped_json_path = run_2d_mapping(paths, GAME_ID)        
    evaluate_mapping(paths["mapped_json"])
        
    ## Seeing the player detection and 2d mapping side by side
    combine_side_by_side(paths, GAME_ID, MODEL_CHOICE)
    #side_by_side2video(paths, GAME_ID, MODEL_CHOICE, 15)