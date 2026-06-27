import cv2
import numpy as np

def create_bullet():
    img = np.zeros((40, 80, 4), dtype=np.uint8)
    # Glow (Cyan/Yellow depending on side, we'll just make it cyan-ish here)
    cv2.ellipse(img, (40, 20), (35, 15), 0, 0, 360, (255, 200, 0, 150), -1) # BGR: Cyan-ish glow
    # Core (White)
    cv2.ellipse(img, (40, 20), (25, 8), 0, 0, 360, (255, 255, 255, 255), -1)
    return img

def create_gun():
    img = np.zeros((100, 150, 4), dtype=np.uint8)
    # Barrel
    cv2.rectangle(img, (30, 40), (110, 65), (200, 200, 200, 255), -1) 
    cv2.rectangle(img, (30, 40), (110, 65), (0, 0, 0, 255), 2)
    # Handle
    cv2.rectangle(img, (90, 40), (120, 90), (250, 100, 50, 255), -1) # Blueish handle
    cv2.rectangle(img, (90, 40), (120, 90), (0, 0, 0, 255), 2)
    # Muzzle/Tip
    cv2.rectangle(img, (10, 45), (30, 60), (50, 50, 50, 255), -1)
    # Accent/Scope
    cv2.rectangle(img, (70, 30), (100, 40), (0, 255, 255, 255), -1) # Yellow accent
    return img

cv2.imwrite('assets/bullet.png', create_bullet())
cv2.imwrite('assets/gun.png', create_gun())
print("Weapon assets created")
