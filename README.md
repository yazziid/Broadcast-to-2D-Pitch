# Automatic Soccer Field Calibration and Player Localization

## Project Overview

This project aims to detectplayers from broadcast video and compare different methods to map into a 2D plan.

## How to Run

Step 1: Create a virtual environment.

```
python -m venv .venv
```

Step 2: Activate the environment

```
source .venv/bin/activate
```

Step 3: Install dependencies

```
pip install -r requirements.txt
```

Step 4: If not downloaded make sure to download the dataset. This can take some time. You also need to unzip the files.

```
python download_data.py
```

Step 5: run the pipeline

```
python main.py
```

## Output

- Visualization comparisons
- Player detection using a fine tuned YOLO8 or rt-DETR model
- 2D player mapping on a canonical pitch
- Key metrics

## Dataset

The dataset can be found following this [link](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025)
