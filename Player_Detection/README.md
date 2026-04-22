## (Optional) Fine Tuning:

We fined tuned the YOLO and RT-DETR model on the SoccerNet data. Due to limited computational power, we reduced the size of the original data and only used the first, middle, and last frame of each game. We also reduced the number of games to 55 games.

If one wishes to fine tune the model, you can do so by running:

```
python yolo_training.py
```

or 

```
python DETR_training.py
```