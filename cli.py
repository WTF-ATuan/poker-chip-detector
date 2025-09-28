#!/usr/bin/env python3
"""
Minimal Poker Chip Analyzer - CLI MVP
Detects poker chips in images, classifies colors, and calculates values.
"""

import argparse
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import math


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess the input image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize if too large (keep aspect ratio)
    height, width = img.shape[:2]
    if width > 1000:
        scale = 1000 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    return img


def detect_circles(img: np.ndarray) -> List[Tuple[int, int, int]]:
    """Detect circles in the image using HoughCircles."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return [(x, y, r) for x, y, r in circles]
    
    return []


def classify_color(img: np.ndarray, x: int, y: int, r: int) -> str:
    """Classify chip color based on HSV average in the circle region."""
    # Create mask for the circle
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Get average HSV values within the circle
    mean_hsv = cv2.mean(hsv, mask=mask)
    h, s, v = mean_hsv[:3]
    
    # Simple color classification based on hue
    if h < 10 or h > 170:  # Red range
        return "red"
    elif 140 <= h <= 170:  # Pink/Magenta range
        return "pink"
    elif 40 <= h <= 80:    # Green range
        return "green"
    else:
        # Default to red for unknown colors
        return "red"


def is_in_roi(x: int, y: int, roi: Tuple[int, int, int, int]) -> bool:
    """Check if a point is within the singles ROI rectangle."""
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_confidence(circles: List[Tuple[int, int, int]]) -> float:
    """Calculate confidence based on circle detection consistency."""
    if not circles:
        return 0.0
    
    # Calculate radius variance as a confidence metric
    radii = [r for _, _, r in circles]
    mean_radius = np.mean(radii)
    radius_variance = np.var(radii)
    
    # Normalize variance (lower variance = higher confidence)
    max_variance = mean_radius * 0.5  # Reasonable threshold
    confidence = max(0.0, min(1.0, 1.0 - (radius_variance / max_variance)))
    
    return round(confidence, 2)


def analyze_chips(image_path: str, denoms: Dict[str, int], per_stack: int, 
                 bb: int, singles_roi: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Main analysis function."""
    # Load and preprocess image
    img = preprocess_image(image_path)
    
    # Detect circles
    circles = detect_circles(img)
    
    if not circles:
        return {
            "stacks": {color: 0 for color in denoms.keys()},
            "singles": {color: 0 for color in denoms.keys()},
            "counts": {color: 0 for color in denoms.keys()},
            "value": 0,
            "bb": 0.0,
            "confidence": 0.0
        }
    
    # Initialize counters
    stacks = {color: 0 for color in denoms.keys()}
    singles = {color: 0 for color in denoms.keys()}
    
    # Process each detected circle
    for x, y, r in circles:
        color = classify_color(img, x, y, r)
        
        if color in denoms:
            if is_in_roi(x, y, singles_roi):
                singles[color] += 1
            else:
                stacks[color] += 1
    
    # Calculate totals
    counts = {}
    total_value = 0
    
    for color in denoms.keys():
        stack_count = stacks[color]
        single_count = singles[color]
        total_count = stack_count * per_stack + single_count
        counts[color] = total_count
        total_value += total_count * denoms[color]
    
    # Calculate BB
    bb_value = total_value / bb if bb > 0 else 0.0
    
    # Calculate confidence
    confidence = calculate_confidence(circles)
    
    return {
        "stacks": stacks,
        "singles": singles,
        "counts": counts,
        "value": total_value,
        "bb": round(bb_value, 1),
        "confidence": confidence
    }


def parse_roi(roi_str: str) -> Tuple[int, int, int, int]:
    """Parse ROI string format: 'x1,y1,x2,y2'."""
    try:
        coords = [int(x.strip()) for x in roi_str.split(',')]
        if len(coords) != 4:
            raise ValueError("ROI must have exactly 4 coordinates")
        return tuple(coords)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid ROI format: {e}")


def main():
    parser = argparse.ArgumentParser(description="Poker Chip Analyzer CLI")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--denoms", required=True, help="JSON string of color denominations")
    parser.add_argument("--per_stack", type=int, required=True, help="Chips per stack")
    parser.add_argument("--bb", type=int, required=True, help="Big blind value")
    parser.add_argument("--singles_roi", required=True, help="Singles ROI as 'x1,y1,x2,y2'")
    
    args = parser.parse_args()
    
    try:
        # Parse denominations
        denoms = json.loads(args.denoms)
        
        # Parse ROI
        roi = parse_roi(args.singles_roi)
        
        # Analyze chips
        result = analyze_chips(
            args.image,
            denoms,
            args.per_stack,
            args.bb,
            roi
        )
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
