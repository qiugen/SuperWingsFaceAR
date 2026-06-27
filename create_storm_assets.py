import cv2
import numpy as np

def create_cloud():
    # Create a dark, stormy cloud
    img = np.zeros((200, 640, 4), dtype=np.uint8)
    
    # Draw overlapping dark gray circles to form a cloud shape at the top
    centers = [(100, 50), (200, 80), (320, 60), (450, 90), (550, 50),
               (150, 20), (280, 20), (400, 30), (500, 20)]
    radii = [60, 70, 80, 75, 65, 50, 60, 55, 50]
    
    for center, r in zip(centers, radii):
        cv2.circle(img, center, r, (100, 100, 100, 220), -1) # Dark gray, mostly opaque
        # Add some lighter highlights
        cv2.circle(img, (center[0]-10, center[1]-10), int(r*0.8), (130, 130, 130, 220), -1)
        
    return img

def create_lightning():
    # Create a lightning bolt
    img = np.zeros((480, 640, 4), dtype=np.uint8)
    
    # Draw a jagged yellow/white line
    points = np.array([
        [320, 80], [280, 200], [330, 200], [260, 350], [300, 350], [220, 480]
    ], np.int32)
    
    # Outer glow
    cv2.polylines(img, [points], False, (0, 255, 255, 100), 15)
    # Inner core
    cv2.polylines(img, [points], False, (255, 255, 255, 255), 5)
    
    return img

cv2.imwrite('assets/cloud.png', create_cloud())
cv2.imwrite('assets/lightning.png', create_lightning())
print("Storm assets created")
