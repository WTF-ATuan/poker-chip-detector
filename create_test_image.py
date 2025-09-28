#!/usr/bin/env python3
"""
Test script for the Poker Chip Analyzer CLI
Creates a simple test image with colored circles to demonstrate functionality
"""

import cv2
import numpy as np
import os

def create_test_image():
    """Create a simple test image with colored circles representing poker chips."""
    # Create a white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Define colors (BGR format for OpenCV)
    red = (0, 0, 255)
    pink = (203, 192, 255)  # Light pink
    green = (0, 255, 0)
    
    # Draw circles representing poker chips
    # Stacks (outside ROI)
    cv2.circle(img, (150, 150), 30, red, -1)
    cv2.circle(img, (200, 150), 30, red, -1)
    cv2.circle(img, (250, 150), 30, red, -1)
    cv2.circle(img, (300, 150), 30, red, -1)
    
    cv2.circle(img, (150, 200), 30, pink, -1)
    cv2.circle(img, (200, 200), 30, pink, -1)
    cv2.circle(img, (250, 200), 30, pink, -1)
    
    # Singles (within ROI: 100,400,300,600)
    cv2.circle(img, (150, 450), 30, pink, -1)
    cv2.circle(img, (200, 450), 30, pink, -1)
    cv2.circle(img, (250, 450), 30, pink, -1)
    cv2.circle(img, (300, 450), 30, pink, -1)
    cv2.circle(img, (150, 500), 30, pink, -1)
    cv2.circle(img, (200, 500), 30, pink, -1)
    cv2.circle(img, (250, 500), 30, pink, -1)
    
    cv2.circle(img, (150, 550), 30, green, -1)
    cv2.circle(img, (200, 550), 30, green, -1)
    cv2.circle(img, (250, 550), 30, green, -1)
    cv2.circle(img, (300, 550), 30, green, -1)
    cv2.circle(img, (150, 580), 30, green, -1)
    cv2.circle(img, (200, 580), 30, green, -1)
    cv2.circle(img, (250, 580), 30, green, -1)
    cv2.circle(img, (300, 580), 30, green, -1)
    cv2.circle(img, (150, 610), 30, green, -1)
    cv2.circle(img, (200, 610), 30, green, -1)
    cv2.circle(img, (250, 610), 30, green, -1)
    cv2.circle(img, (300, 610), 30, green, -1)
    
    # Draw ROI rectangle for reference
    cv2.rectangle(img, (100, 400), (300, 600), (0, 0, 0), 2)
    
    return img

def main():
    """Create test image and save it."""
    # Create test image
    test_img = create_test_image()
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save test image
    cv2.imwrite('data/example1.jpg', test_img)
    print("Test image created: data/example1.jpg")
    print("\nYou can now test the CLI with:")
    print('python cli.py --image data/example1.jpg --denoms \'{"red":5000,"pink":1000,"green":500}\' --per_stack 20 --bb 2000 --singles_roi "100,400,300,600"')

if __name__ == "__main__":
    main()
