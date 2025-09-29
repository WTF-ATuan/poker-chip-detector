# Poker Chip Analyzer CLI

A minimal Python CLI tool that analyzes poker chip images to detect circles, classify colors, and calculate total values and big blinds.

## Features

- **Circle Detection**: Uses OpenCV's HoughCircles to detect poker chips
- **Color Classification**: Classifies chips into red, pink, and green based on HSV color space
- **ROI Separation**: Distinguishes between singles (within ROI) and stacks (outside ROI)
- **Value Calculation**: Computes total chip value and big blind equivalent
- **Confidence Scoring**: Provides detection confidence based on circle consistency

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd poker-chip-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1) 產生偽標註與可視化
source .venv311/bin/activate
python bootstrap_zero_shot.py

# 2) 建立 YOLO 資料集結構
python prepare_yolo_dataset.py

# 3) 訓練 YOLOv8n
yolo detect train model=yolov8n.pt data=data/yolo/dataset.yaml imgsz=640 epochs=60 batch=8 workers=0 amp=False

# 4) 預測可視化
yolo detect predict model=runs/detect/train/weights/best.pt source=data/yolo/val/images save
```

### Parameters

- `--image`: Path to the input image containing poker chips
- `--denoms`: JSON string mapping color names to denomination values
- `--per_stack`: Number of chips per stack
- `--bb`: Big blind value for BB calculation
- `--singles_roi`: ROI rectangle as "x1,y1,x2,y2" for singles detection

### Output Format

```json
{
  "stacks": {"red": 4, "pink": 3, "green": 0},
  "singles": {"red": 0, "pink": 7, "green": 12},
  "counts": {"red": 80, "pink": 67, "green": 12},
  "value": 487000,
  "bb": 243.5,
  "confidence": 0.86
}
```

## How It Works

1. **Image Preprocessing**: Loads and resizes the input image if needed
2. **Circle Detection**: Uses HoughCircles to detect circular chip shapes
3. **Color Classification**: Analyzes HSV values within each detected circle
4. **ROI Filtering**: Separates singles (within ROI) from stacks (outside ROI)
5. **Value Calculation**: Computes total value and big blind equivalent
6. **Confidence Scoring**: Calculates detection confidence based on circle consistency

## Color Classification

The tool classifies chips based on HSV hue values:
- **Red**: Hue < 10° or > 170°
- **Pink**: Hue 140° - 170°
- **Green**: Hue 40° - 80°

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- NumPy

## Sample Data

Place your poker chip images in the `data/` directory. The tool expects images with clear, circular poker chips that can be detected using computer vision techniques.

## Limitations

- Color classification is based on simple HSV thresholds
- Circle detection may struggle with overlapping chips
- ROI coordinates must be manually specified
- Best results with well-lit, high-contrast images

## Contributing

This is a minimal MVP implementation. Potential improvements:
- Machine learning-based color classification
- Automatic ROI detection
- Support for more chip colors
- Better handling of overlapping chips
- GUI interface
