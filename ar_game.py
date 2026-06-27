import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh and Pose
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load Assets
jett_mask = cv2.imread('assets/jett_mask.png', cv2.IMREAD_UNCHANGED)
dizzy_mask = cv2.imread('assets/dizzy_mask.png', cv2.IMREAD_UNCHANGED)
rain_img = cv2.imread('assets/rain.png', cv2.IMREAD_UNCHANGED)
sun_img = cv2.imread('assets/sun.png', cv2.IMREAD_UNCHANGED)
cloud_img = cv2.imread('assets/cloud.png', cv2.IMREAD_UNCHANGED)
lightning_img = cv2.imread('assets/lightning.png', cv2.IMREAD_UNCHANGED)

current_character = "Jett (乐迪)"
mask_img = jett_mask

def overlay_transparent(background, overlay, x, y):
    """Overlays a transparent PNG onto a background image at (x,y)."""
    bg_h, bg_w, bg_channels = background.shape
    ol_h, ol_w, ol_channels = overlay.shape

    if x >= bg_w or y >= bg_h:
        return background

    h, w = min(ol_h, bg_h - y), min(ol_w, bg_w - x)
    
    if x < 0:
        overlay = overlay[:, -x:]
        w += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h += y
        y = 0
        
    if h <= 0 or w <= 0:
        return background

    overlay_image = overlay[:h, :w, :3]
    mask = overlay[:h, :w, 3] / 255.0
    
    for c in range(3):
        background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - mask) + overlay_image[:, :, c] * mask

    return background

def is_smiling(landmarks, w, h):
    """Simple smile detection using mouth corner distance vs face width."""
    # Mouth corners: 61 (left), 291 (right)
    # Face width: 234 (right edge), 93 (left edge)
    mouth_width = np.linalg.norm(
        np.array([landmarks[291].x * w, landmarks[291].y * h]) - 
        np.array([landmarks[61].x * w, landmarks[61].y * h])
    )
    face_width = np.linalg.norm(
        np.array([landmarks[234].x * w, landmarks[234].y * h]) - 
        np.array([landmarks[93].x * w, landmarks[93].y * h])
    )
    # Ratio usually > 0.45 when smiling
    return (mouth_width / face_width) > 0.45

def is_shielding_rain(pose_landmarks):
    """Detect if hands are raised to head level (shielding rain gesture)."""
    if not pose_landmarks:
        return False
        
    landmarks = pose_landmarks.landmark
    
    # Get y-coordinates (lower value = higher on screen)
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    
    # Check if visibility of wrists is high enough
    left_visible = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.5
    right_visible = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > 0.5
    
    # If either wrist is raised near or above the nose level, trigger rain
    # (nose_y is typically around 0.3-0.5. If wrist_y is smaller, it means hands are up)
    if left_visible and (left_wrist_y < nose_y + 0.1):
        return True
    if right_visible and (right_wrist_y < nose_y + 0.1):
        return True
        
    return False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

rain_offset_y = 0
lightning_timer = 0
lightning_visible = False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(image_rgb)
    results_pose = pose.process(image_rgb)

    current_state = "Neutral"

    # 1. Determine Expression / Gesture
    if is_shielding_rain(results_pose.pose_landmarks):
        current_state = "Shielding (Storm)"
        
        # Apply Lightning (Flashes randomly)
        lightning_timer += 1
        if lightning_timer > np.random.randint(10, 30):
            lightning_visible = not lightning_visible
            lightning_timer = 0
            
        if lightning_visible and lightning_img is not None:
            lightning_resized = cv2.resize(lightning_img, (w, h))
            image = overlay_transparent(image, lightning_resized, 0, 0)
            
        # Apply Rain Effect
        if rain_img is not None:
            rain_resized = cv2.resize(rain_img, (w, h))
            rain_offset_y = (rain_offset_y + 20) % h # Faster rain
            rain_top = rain_resized[h-rain_offset_y:, :]
            rain_bottom = rain_resized[:h-rain_offset_y, :]
            image = overlay_transparent(image, rain_top, 0, 0)
            image = overlay_transparent(image, rain_bottom, 0, rain_offset_y)
            
        # Apply Dark Clouds at the top
        if cloud_img is not None:
            cloud_resized = cv2.resize(cloud_img, (w, int(h * 0.4)))
            image = overlay_transparent(image, cloud_resized, 0, 0)

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Check smiling (only if not already raining)
            if current_state == "Neutral" and is_smiling(landmarks, w, h):
                current_state = "Smiling (Sun)"
                if sun_img is not None:
                    sun_resized = cv2.resize(sun_img, (w, h))
                    image = overlay_transparent(image, sun_resized, 0, 0)

            # 2. Apply Mask
            if mask_img is not None:
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                
                face_x = int(min(x_coords))
                face_y = int(min(y_coords))
                face_w = int(max(x_coords) - face_x)
                
                top_head = (int(landmarks[10].x * w), int(landmarks[10].y * h))
                
                mask_w = int(face_w * 1.5)
                mask_h = int(mask_img.shape[0] * (mask_w / mask_img.shape[1]))
                
                mask_resized = cv2.resize(mask_img, (mask_w, mask_h))
                
                pos_x = int(landmarks[1].x * w) - mask_w // 2 
                pos_y = top_head[1] - mask_h // 2 
                
                image = overlay_transparent(image, mask_resized, pos_x, pos_y)

    cv2.putText(image, f"Expression: {current_state}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(image, f"Character: {current_character}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2, cv2.LINE_AA)
    
    cv2.putText(image, "Press 'SPACE' to switch character", (20, h - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Press 'Q' to quit", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Super Wings AR Game', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if current_character == "Jett (乐迪)":
            current_character = "Dizzy (小爱)"
            mask_img = dizzy_mask
        else:
            current_character = "Jett (乐迪)"
            mask_img = jett_mask

cap.release()
cv2.destroyAllWindows()
