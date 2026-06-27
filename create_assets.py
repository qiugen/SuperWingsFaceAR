import cv2
import numpy as np

# Create Jett face mask (red helmet/goggles placeholder)
jett_mask = np.zeros((400, 400, 4), dtype=np.uint8)
# Red helmet top
cv2.ellipse(jett_mask, (200, 200), (180, 150), 0, 180, 360, (0, 0, 255, 200), -1)
# White goggles outline
cv2.ellipse(jett_mask, (120, 180), (50, 40), 0, 0, 360, (255, 255, 255, 255), 5)
cv2.ellipse(jett_mask, (280, 180), (50, 40), 0, 0, 360, (255, 255, 255, 255), 5)
cv2.imwrite('assets/jett_mask.png', jett_mask)

# Create Rain effect placeholder
rain_effect = np.zeros((480, 640, 4), dtype=np.uint8)
# Draw random blue lines for rain
for _ in range(200):
    x = np.random.randint(0, 640)
    y = np.random.randint(0, 480)
    cv2.line(rain_effect, (x, y), (x-10, y+30), (255, 200, 100, 150), 2)
cv2.imwrite('assets/rain.png', rain_effect)

# Create Sun effect placeholder
sun_effect = np.zeros((480, 640, 4), dtype=np.uint8)
# Draw a sun in top right corner
cv2.circle(sun_effect, (540, 100), 60, (0, 255, 255, 255), -1)
for i in range(0, 360, 30):
    start_x = int(540 + 70 * np.cos(np.radians(i)))
    start_y = int(100 + 70 * np.sin(np.radians(i)))
    end_x = int(540 + 100 * np.cos(np.radians(i)))
    end_y = int(100 + 100 * np.sin(np.radians(i)))
    cv2.line(sun_effect, (start_x, start_y), (end_x, end_y), (0, 255, 255, 255), 5)
cv2.imwrite('assets/sun.png', sun_effect)

print("Assets created successfully.")
