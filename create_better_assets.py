import cv2
import numpy as np

def create_jett():
    img = np.zeros((400, 400, 4), dtype=np.uint8)
    # Red main helmet
    cv2.ellipse(img, (200, 200), (180, 160), 0, 180, 360, (40, 40, 220, 255), -1) # Red
    # Helmet wings/ears
    cv2.ellipse(img, (50, 150), (40, 80), 30, 0, 360, (40, 40, 220, 255), -1)
    cv2.ellipse(img, (350, 150), (40, 80), -30, 0, 360, (40, 40, 220, 255), -1)
    # White goggles area
    cv2.ellipse(img, (200, 200), (140, 60), 0, 180, 360, (255, 255, 255, 255), -1)
    # Blue eyes
    cv2.ellipse(img, (140, 180), (30, 40), 0, 0, 360, (255, 100, 50, 255), -1)
    cv2.ellipse(img, (260, 180), (30, 40), 0, 0, 360, (255, 100, 50, 255), -1)
    # Black pupils
    cv2.ellipse(img, (140, 180), (15, 20), 0, 0, 360, (0, 0, 0, 255), -1)
    cv2.ellipse(img, (260, 180), (15, 20), 0, 0, 360, (0, 0, 0, 255), -1)
    # White highlights
    cv2.circle(img, (145, 175), 5, (255, 255, 255, 255), -1)
    cv2.circle(img, (265, 175), 5, (255, 255, 255, 255), -1)
    return img

def create_dizzy():
    img = np.zeros((400, 400, 4), dtype=np.uint8)
    # Pink/Magenta main helmet
    cv2.ellipse(img, (200, 200), (180, 160), 0, 180, 360, (200, 100, 255, 255), -1) # Pink
    # Rescue light on top
    cv2.ellipse(img, (200, 40), (30, 40), 0, 180, 360, (0, 255, 255, 255), -1) # Yellow
    # White goggles area
    cv2.ellipse(img, (200, 200), (140, 60), 0, 180, 360, (255, 255, 255, 255), -1)
    # Green/Teal eyes
    cv2.ellipse(img, (140, 180), (30, 40), 0, 0, 360, (150, 255, 50, 255), -1)
    cv2.ellipse(img, (260, 180), (30, 40), 0, 0, 360, (150, 255, 50, 255), -1)
    # Black pupils
    cv2.ellipse(img, (140, 180), (15, 20), 0, 0, 360, (0, 0, 0, 255), -1)
    cv2.ellipse(img, (260, 180), (15, 20), 0, 0, 360, (0, 0, 0, 255), -1)
    # White highlights
    cv2.circle(img, (145, 175), 5, (255, 255, 255, 255), -1)
    cv2.circle(img, (265, 175), 5, (255, 255, 255, 255), -1)
    return img

cv2.imwrite('assets/jett_mask.png', create_jett())
cv2.imwrite('assets/dizzy_mask.png', create_dizzy())
print("Created new masks")
