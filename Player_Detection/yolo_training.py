from ultralytics import YOLO

# We fine tune the original yolo model
model = YOLO('yolov8m.pt') 

results = model.train(
    data='data.yaml',      
    epochs=20,             
    imgsz=640,             
    batch=16,              
    name='soccernet_m',
    cache = True    
)

print("Training complete! The best weights are saved in runs/detect/soccernet_m/weights/best.pt")