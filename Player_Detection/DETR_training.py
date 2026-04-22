from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt') 

results = model.train(
    data='data.yaml',      
    epochs=20,            
    imgsz=640,             
    batch=4,              
    name='soccernet_rtdetr', 
    cache=True
)

print("Training complete! Your RT-DETR weights are in runs/detect/soccernet_rtdetr/weights/best.pt")