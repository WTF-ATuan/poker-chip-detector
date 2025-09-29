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
import os


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


def detect_chip_stacks(img: np.ndarray, color_name: str) -> List[Tuple[int, int, int, int]]:
    """
    Detect chip stacks using contour analysis instead of circle detection.
    Returns list of (x, y, width, height) bounding boxes for chip stacks.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for different chip colors (more restrictive)
    color_ranges = {
        'red': {
            'lower': [(0, 100, 100), (160, 100, 100)],  # More restrictive red
            'upper': [(10, 255, 255), (180, 255, 255)]
        },
        'purple': {
            'lower': [(120, 80, 80)],  # Lower saturation threshold for purple
            'upper': [(150, 255, 255)]
        },
        'pink': {
            'lower': [(145, 100, 100)],
            'upper': [(165, 255, 255)]
        },
        'green': {
            'lower': [(45, 100, 100)],
            'upper': [(75, 255, 255)]
        }
    }
    
    if color_name not in color_ranges:
        return []
    
    ranges = color_ranges[color_name]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Create mask for this color
    for lower, upper in zip(ranges['lower'], ranges['upper']):
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask += cv2.inRange(hsv, lower_np, upper_np)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chip_stacks = []
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Filter by area (too small or too large are likely not chip stacks)
        if area < 500 or area > 50000:  # Adjust these thresholds as needed
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (chip stacks should be roughly circular/square)
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
        
        # Calculate circularity (how close to a circle)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Only accept reasonably circular shapes
        if circularity > 0.3:  # Adjust threshold as needed
            chip_stacks.append((x, y, w, h))
    
    return chip_stacks


def create_chip_stack_debug_image(img: np.ndarray, stacks_by_color: Dict[str, List[Tuple[int, int, int, int]]], 
                                  roi: Tuple[int, int, int, int], output_path: str):
    """Create debug visualization showing detected chip stacks."""
    debug_img = img.copy()
    
    # Define colors for visualization (BGR format)
    color_colors = {
        'red': (0, 0, 255),      # Red
        'purple': (128, 0, 128),  # Purple
        'pink': (203, 192, 255),  # Light pink
        'green': (0, 255, 0)      # Green
    }
    
    # Draw ROI rectangle
    x1, y1, x2, y2 = roi
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(debug_img, "ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw chip stacks for each color
    for color, stacks in stacks_by_color.items():
        if stacks:
            stack_color = color_colors.get(color, (255, 255, 255))
            
            for i, (x, y, w, h) in enumerate(stacks):
                # Draw bounding rectangle
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), stack_color, 2)
                
                # Draw label
                label = f"{color} stack"
                cv2.putText(debug_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stack_color, 2)
                
                # Draw center point
                center_x, center_y = x + w//2, y + h//2
                cv2.circle(debug_img, (center_x, center_y), 5, stack_color, -1)
    
    # Save debug image
    cv2.imwrite(output_path, debug_img)
    print(f"Debug image saved: {output_path}")


def is_in_roi(x: int, y: int, roi: Tuple[int, int, int, int]) -> bool:
    """Check if a point is within the singles ROI rectangle."""
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_confidence(circles_by_color: Dict[str, List[Tuple[int, int, int]]]) -> float:
    """Calculate confidence based on detection consistency across colors."""
    if not circles_by_color:
        return 0.0
    
    total_circles = sum(len(circles) for circles in circles_by_color.values())
    if total_circles == 0:
        return 0.0
    
    # Calculate radius consistency within each color
    radius_consistencies = []
    for color, circles in circles_by_color.items():
        if len(circles) > 1:
            radii = [r for _, _, r in circles]
            mean_radius = np.mean(radii)
            radius_variance = np.var(radii)
            consistency = max(0, 1 - (radius_variance / (mean_radius * 0.3)))
            radius_consistencies.append(consistency)
    
    # Overall confidence based on detection count and consistency
    detection_confidence = min(1.0, total_circles / 50)  # Normalize by expected chip count
    consistency_confidence = np.mean(radius_consistencies) if radius_consistencies else 0.5
    
    return round((detection_confidence + consistency_confidence) / 2, 2)


def analyze_chips(image_path: str, denoms: Dict[str, int], per_stack: int, 
                 bb: int, singles_roi: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Improved analysis using chip stack detection instead of individual circles."""
    # Load and preprocess image
    img = preprocess_image(image_path)
    
    # Initialize counters
    stacks = {color: 0 for color in denoms.keys()}
    singles = {color: 0 for color in denoms.keys()}
    stacks_by_color = {}
    
    # Process each color separately
    for color in denoms.keys():
        # Detect chip stacks for this color
        chip_stacks = detect_chip_stacks(img, color)
        stacks_by_color[color] = chip_stacks
        
        # Count stacks vs singles
        for x, y, w, h in chip_stacks:
            center_x, center_y = x + w//2, y + h//2
            if is_in_roi(center_x, center_y, singles_roi):
                singles[color] += 1
            else:
                stacks[color] += 1
    
    # Create debug image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    debug_path = f"data/debug/{base_name}_debug.jpg"
    create_chip_stack_debug_image(img, stacks_by_color, singles_roi, debug_path)
    
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
    
    # Calculate confidence based on detected stacks
    total_stacks = sum(len(stacks) for stacks in stacks_by_color.values())
    confidence = min(1.0, total_stacks / 10) if total_stacks > 0 else 0.0
    
    return {
        "stacks": stacks,
        "singles": singles,
        "counts": counts,
        "value": total_value,
        "bb": round(bb_value, 1),
        "confidence": round(confidence, 2),
        "detected_stacks": {color: len(stacks) for color, stacks in stacks_by_color.items()},
        "debug_image": debug_path
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
