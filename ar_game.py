import os
import random
import time

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")


def load_asset(name):
    return cv2.imread(os.path.join(ASSET_DIR, name), cv2.IMREAD_UNCHANGED)


# 素材：全部是 PNG 透明图，缺失也不会导致游戏崩溃。
jett_mask = load_asset("jett_mask.png")
dizzy_mask = load_asset("dizzy_mask.png")
rain_img = load_asset("rain.png")
sun_img = load_asset("sun.png")
cloud_img = load_asset("cloud.png")
lightning_img = load_asset("lightning.png")
bullet_img = load_asset("bullet.png")
gun_img = load_asset("gun.png")
grenade_img = load_asset("grenade.png")


# OpenCV 自带的人脸/笑脸检测，不再依赖 MediaPipe，运行更简单。
face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)
smile_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml")
)


left_character = "Dizzy"
right_character = "Jett"
storm_manual_until = 0
sun_manual_until = 0
rain_offset_y = 0
lightning_timer = 0
lightning_visible = False
prev_gray = None
bullets = []
grenades = []
hit_effects = []
explosions = []
gesture_cooldowns = {
    "left_bullet": 0,
    "right_bullet": 0,
    "left_grenade": 0,
    "right_grenade": 0,
}
gesture_messages = []


def overlay_transparent(background, overlay, x, y):
    """把透明 PNG 叠到摄像头画面上。"""
    if overlay is None:
        return background
    if overlay.shape[2] < 4:
        return background

    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return background

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + ol_w, bg_w)
    y2 = min(y + ol_h, bg_h)

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    alpha = overlay_crop[:, :, 3:4].astype(float) / 255.0
    background[y1:y2, x1:x2] = (
        background[y1:y2, x1:x2].astype(float) * (1 - alpha)
        + overlay_crop[:, :, :3].astype(float) * alpha
    ).astype(np.uint8)
    return background


def get_mask_for_character(char_name):
    return jett_mask if char_name == "Jett" else dizzy_mask


def switch_character(char_name):
    return "Jett" if char_name == "Dizzy" else "Dizzy"


def detect_faces(gray):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    # 大脸优先，最多保留 6 个人，够多人对战用。
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:6]
    return faces


def face_is_smiling(gray, face):
    x, y, fw, fh = face
    roi = gray[y : y + fh, x : x + fw]
    smiles = smile_cascade.detectMultiScale(
        roi,
        scaleFactor=1.7,
        minNeighbors=20,
        minSize=(max(25, fw // 5), max(12, fh // 12)),
    )
    return len(smiles) > 0


def get_motion_mask(gray):
    """基于前后帧差分生成动作区域，用于体感控制。"""
    global prev_gray
    if prev_gray is None:
        prev_gray = gray.copy()
        return np.zeros_like(gray)

    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.medianBlur(motion_mask, 5)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)
    prev_gray = gray.copy()
    return motion_mask


def motion_ratio(motion_mask, left, top, right, bottom):
    """计算某个区域里的动作占比。"""
    h, w = motion_mask.shape[:2]
    left = max(0, min(w, int(left)))
    right = max(0, min(w, int(right)))
    top = max(0, min(h, int(top)))
    bottom = max(0, min(h, int(bottom)))
    if right <= left or bottom <= top:
        return 0
    roi = motion_mask[top:bottom, left:right]
    if roi.size == 0:
        return 0
    return cv2.countNonZero(roi) / roi.size


def detect_shielding_motion(motion_mask, faces):
    """简单实用版“挡雨手势”：脸上方出现明显运动，就认为抬手挡雨。"""
    if not faces:
        return False

    for x, y, fw, fh in faces:
        top = y - int(fh * 0.9)
        bottom = y + int(fh * 0.25)
        left = x - int(fw * 0.5)
        right = x + int(fw * 1.5)
        if motion_ratio(motion_mask, left, top, right, bottom) > 0.08:
            return True
    return False


def apply_storm(image):
    global rain_offset_y, lightning_timer, lightning_visible
    h, w = image.shape[:2]

    if cloud_img is not None:
        cloud_resized = cv2.resize(cloud_img, (w, int(h * 0.38)))
        image = overlay_transparent(image, cloud_resized, 0, 0)

    if rain_img is not None:
        rain_resized = cv2.resize(rain_img, (w, h))
        rain_offset_y = (rain_offset_y + 22) % h
        image = overlay_transparent(image, rain_resized[h - rain_offset_y :, :], 0, 0)
        image = overlay_transparent(image, rain_resized[: h - rain_offset_y, :], 0, rain_offset_y)

    lightning_timer += 1
    if lightning_timer > random.randint(8, 22):
        lightning_visible = not lightning_visible
        lightning_timer = 0
    if lightning_visible and lightning_img is not None:
        lightning_resized = cv2.resize(lightning_img, (w, h))
        image = overlay_transparent(image, lightning_resized, 0, 0)

    return image


def draw_player(image, face, side, char_name):
    x, y, fw, fh = face
    cx = x + fw // 2
    cy = y + fh // 2

    mask = get_mask_for_character(char_name)
    if mask is not None and fw > 0:
        mask_w = int(fw * 1.65)
        mask_h = int(mask.shape[0] * (mask_w / mask.shape[1]))
        mask_resized = cv2.resize(mask, (mask_w, mask_h))
        image = overlay_transparent(image, mask_resized, cx - mask_w // 2, y - mask_h // 3)

    label = "L" if side == "left" else "R"
    cv2.putText(image, label, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return image


def draw_guns(image, left_faces, right_faces):
    if gun_img is None:
        return image
    gun_h, gun_w = gun_img.shape[:2]
    # 当前 gun.png 默认枪口朝左；左边玩家要朝右，所以使用水平翻转版。
    gun_right = cv2.flip(gun_img, 1)
    gun_left = gun_img
    for face in left_faces:
        x, y, fw, fh = face
        image = overlay_transparent(image, gun_right, x + fw, y + fh // 2 - gun_h // 2)

    for face in right_faces:
        x, y, fw, fh = face
        image = overlay_transparent(image, gun_left, x - gun_w, y + fh // 2 - gun_h // 2)
    return image


def shoot_from_faces(side, faces):
    for x, y, fw, fh in faces:
        if side == "left":
            bullets.append({"x": x + fw + 18, "y": y + fh // 2, "dx": 28, "side": "left"})
        else:
            bullets.append({"x": x - 18, "y": y + fh // 2, "dx": -28, "side": "right"})


def throw_grenade_from_faces(side, faces):
    """从玩家脸旁投掷手雷：有横向速度、向上初速度和重力，形成抛物线。"""
    for x, y, fw, fh in faces:
        if side == "left":
            grenades.append(
                {
                    "x": float(x + fw + 8),
                    "y": float(y + fh * 0.45),
                    "vx": 13.5,
                    "vy": -18.0,
                    "life": 58,
                    "side": "left",
                    "trail": [],
                    "angle": 0,
                }
            )
        else:
            grenades.append(
                {
                    "x": float(x - 8),
                    "y": float(y + fh * 0.45),
                    "vx": -13.5,
                    "vy": -18.0,
                    "life": 58,
                    "side": "right",
                    "trail": [],
                    "angle": 0,
                }
            )


def add_gesture_message(text, x, y, color):
    gesture_messages.append({"text": text, "x": int(x), "y": int(y), "timer": 28, "color": color})


def draw_gesture_guides(image, face, side):
    """画出简单体感识别区域，方便站位：脸两侧挥拳/握拳，下半身抬腿。"""
    x, y, fw, fh = face
    h, w = image.shape[:2]
    hand_top = y + int(fh * 0.15)
    hand_bottom = y + int(fh * 1.65)
    left_hand = (x - int(fw * 1.15), hand_top, x - int(fw * 0.08), hand_bottom)
    right_hand = (x + int(fw * 1.08), hand_top, x + int(fw * 2.15), hand_bottom)
    leg_box = (x - int(fw * 0.85), y + int(fh * 1.65), x + int(fw * 1.85), min(h, y + int(fh * 4.4)))

    for box, color in [(left_hand, (255, 220, 0)), (right_hand, (255, 220, 0)), (leg_box, (0, 180, 255))]:
        l, t, r, b = box
        l, r = max(0, l), min(w, r)
        t, b = max(0, t), min(h, b)
        if r > l and b > t:
            cv2.rectangle(image, (l, t), (r, b), color, 1, cv2.LINE_AA)

    return image


def process_body_gestures(image, motion_mask, left_faces, right_faces):
    """体感控制：双手握拳/出拳发射子弹；腿部动作投掷手雷。"""
    now = time.time()
    h, w = image.shape[:2]

    for side, faces in [("left", left_faces), ("right", right_faces)]:
        for face in faces:
            x, y, fw, fh = face
            image = draw_gesture_guides(image, face, side)

            hand_top = y + int(fh * 0.15)
            hand_bottom = y + int(fh * 1.65)
            left_hand_ratio = motion_ratio(
                motion_mask,
                x - fw * 1.15,
                hand_top,
                x - fw * 0.08,
                hand_bottom,
            )
            right_hand_ratio = motion_ratio(
                motion_mask,
                x + fw * 1.08,
                hand_top,
                x + fw * 2.15,
                hand_bottom,
            )

            # 双手握拳/向前打拳：脸两侧手部区域同时出现动作，就自动发射子弹。
            bullet_key = f"{side}_bullet"
            if left_hand_ratio > 0.035 and right_hand_ratio > 0.035 and now > gesture_cooldowns[bullet_key]:
                shoot_from_faces(side, [face])
                gesture_cooldowns[bullet_key] = now + 0.9
                add_gesture_message(f"{side.upper()} FIST BULLET!", x, max(28, y - 32), (0, 255, 255))

            leg_ratio = motion_ratio(
                motion_mask,
                x - fw * 0.85,
                y + fh * 1.65,
                x + fw * 1.85,
                y + fh * 4.4,
            )

            # 如果摄像头只拍到上半身，下半身区域很小，则额外用本侧屏幕下半部分作为腿部动作兜底。
            if leg_ratio < 0.01:
                if side == "left":
                    leg_ratio = motion_ratio(motion_mask, 0, h * 0.55, w * 0.5, h)
                else:
                    leg_ratio = motion_ratio(motion_mask, w * 0.5, h * 0.55, w, h)

            grenade_key = f"{side}_grenade"
            if leg_ratio > 0.055 and now > gesture_cooldowns[grenade_key]:
                throw_grenade_from_faces(side, [face])
                gesture_cooldowns[grenade_key] = now + 1.5
                add_gesture_message(f"{side.upper()} LEG GRENADE!", x, y + fh + 28, (0, 180, 255))

    return image


def add_explosion(x, y, side, left_faces, right_faces):
    """添加大爆炸，并检查爆炸范围内是否命中对方。"""
    radius = 120
    explosions.append({"x": int(x), "y": int(y), "timer": 24, "max_radius": radius})

    targets = right_faces if side == "left" else left_faces
    for tx, ty, fw, fh in targets:
        cx = tx + fw // 2
        cy = ty + fh // 2
        if np.hypot(x - cx, y - cy) < radius:
            hit_effects.append({"x": cx, "y": cy, "timer": 18})


def draw_rotated_png(image, overlay, center_x, center_y, size, angle):
    """绘制会旋转的透明 PNG，用在飞行中的手雷上。"""
    if overlay is None:
        cv2.circle(image, (int(center_x), int(center_y)), size // 2, (60, 160, 70), -1)
        cv2.circle(image, (int(center_x), int(center_y)), size // 2, (255, 255, 255), 2)
        return image

    resized = cv2.resize(overlay, (size, size))
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        resized,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return overlay_transparent(image, rotated, int(center_x - size / 2), int(center_y - size / 2))


def process_grenades(image, left_faces, right_faces):
    """更新手雷飞行、画抛物线轨迹、触发爆炸。"""
    h, w = image.shape[:2]
    gravity = 0.85

    for grenade in grenades[:]:
        grenade["trail"].append((int(grenade["x"]), int(grenade["y"])))
        grenade["trail"] = grenade["trail"][-18:]

        grenade["x"] += grenade["vx"]
        grenade["y"] += grenade["vy"]
        grenade["vy"] += gravity
        grenade["life"] -= 1
        grenade["angle"] += 18 if grenade["side"] == "left" else -18

        for i in range(1, len(grenade["trail"])):
            thickness = max(1, i // 4 + 1)
            cv2.line(
                image,
                grenade["trail"][i - 1],
                grenade["trail"][i],
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        cv2.circle(image, (int(grenade["x"]), int(grenade["y"])), 28, (255, 255, 255), 2)
        image = draw_rotated_png(image, grenade_img, grenade["x"], grenade["y"], 58, grenade["angle"])

        targets = right_faces if grenade["side"] == "left" else left_faces
        hit_target = False
        for x, y, fw, fh in targets:
            cx = x + fw // 2
            cy = y + fh // 2
            if np.hypot(grenade["x"] - cx, grenade["y"] - cy) < max(fw, fh) * 0.75:
                hit_target = True
                break

        should_explode = (
            hit_target
            or grenade["life"] <= 0
            or grenade["y"] > h - 45
            or grenade["x"] < -60
            or grenade["x"] > w + 60
        )
        if should_explode:
            add_explosion(grenade["x"], min(grenade["y"], h - 45), grenade["side"], left_faces, right_faces)
            grenades.remove(grenade)

    return image


def process_explosions(image):
    """绘制手雷的大爆炸效果：火球 + 冲击波。"""
    for explosion in explosions[:]:
        progress = 1 - explosion["timer"] / 24
        radius = int(18 + explosion["max_radius"] * progress)
        x = explosion["x"]
        y = explosion["y"]

        cv2.circle(image, (x, y), radius, (0, 90, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), int(radius * 0.68), (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), int(radius * 0.38), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius + 8, (255, 255, 255), 3, cv2.LINE_AA)

        for angle in range(0, 360, 45):
            rad = np.deg2rad(angle)
            end_x = int(x + np.cos(rad) * (radius + 28))
            end_y = int(y + np.sin(rad) * (radius + 28))
            cv2.line(image, (x, y), (end_x, end_y), (0, 255, 255), 3, cv2.LINE_AA)

        explosion["timer"] -= 1
        if explosion["timer"] <= 0:
            explosions.remove(explosion)

    return image


def process_bullets(image, left_faces, right_faces):
    h, w = image.shape[:2]
    # 当前 bullet.png 默认朝左；左边玩家向右射击，所以左侧子弹使用翻转版。
    bullet_left = cv2.flip(bullet_img, 1) if bullet_img is not None else None
    bullet_right = bullet_img

    for bullet in bullets[:]:
        old_x = bullet["x"]
        bullet["x"] += bullet["dx"]
        if bullet["x"] < -50 or bullet["x"] > w + 50:
            bullets.remove(bullet)
            continue

        # 先画一层高亮弹道和光圈，保证子弹在摄像头画面里始终可见。
        color = (0, 255, 255) if bullet["side"] == "left" else (0, 150, 255)
        cv2.line(
            image,
            (int(old_x), int(bullet["y"])),
            (int(bullet["x"]), int(bullet["y"])),
            color,
            6,
            cv2.LINE_AA,
        )
        cv2.circle(image, (int(bullet["x"]), int(bullet["y"])), 16, (255, 255, 255), -1)
        cv2.circle(image, (int(bullet["x"]), int(bullet["y"])), 11, color, -1)

        img = bullet_left if bullet["side"] == "left" else bullet_right
        if img is not None:
            # 放大一点，避免原图太小看不清。
            bullet_w = 64
            bullet_h = max(24, int(img.shape[0] * (bullet_w / img.shape[1])))
            img_resized = cv2.resize(img, (bullet_w, bullet_h))
            image = overlay_transparent(
                image,
                img_resized,
                int(bullet["x"] - bullet_w // 2),
                int(bullet["y"] - bullet_h // 2),
            )
        else:
            cv2.circle(image, (int(bullet["x"]), int(bullet["y"])), 12, (0, 255, 255), -1)

        targets = right_faces if bullet["side"] == "left" else left_faces
        for x, y, fw, fh in targets:
            cx = x + fw // 2
            cy = y + fh // 2
            if np.hypot(bullet["x"] - cx, bullet["y"] - cy) < max(fw, fh) * 0.55:
                hit_effects.append({"x": cx, "y": cy, "timer": 16})
                bullets.remove(bullet)
                break

    for effect in hit_effects[:]:
        radius = max(8, (17 - effect["timer"]) * 7)
        cv2.circle(image, (effect["x"], effect["y"]), radius, (0, 140, 255), -1)
        cv2.circle(image, (effect["x"], effect["y"]), max(4, int(radius * 0.55)), (0, 255, 255), -1)
        effect["timer"] -= 1
        if effect["timer"] <= 0:
            hit_effects.remove(effect)
    return image


def draw_gesture_messages(image):
    for msg in gesture_messages[:]:
        cv2.putText(
            image,
            msg["text"],
            (msg["x"], msg["y"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            msg["color"],
            2,
            cv2.LINE_AA,
        )
        msg["timer"] -= 1
        if msg["timer"] <= 0:
            gesture_messages.remove(msg)
    return image


def draw_ui(image, state, left_faces, right_faces):
    h, w = image.shape[:2]
    cv2.line(image, (w // 2, 0), (w // 2, h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"State: {state}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"L:{left_character} ({len(left_faces)})  R:{right_character} ({len(right_faces)})", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 100), 2)
    cv2.putText(image, "Body: both fists = gun | leg/kick = grenade", (20, h - 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(image, "A:Left role | Enter:Right role | Space:Swap | R:Storm | S:Sun", (20, h - 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(image, "Z:Left gun | M:Right gun | X:Left grenade | N:Right grenade", (20, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(image, "Q:Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return image


def main():
    global left_character, right_character, storm_manual_until, sun_manual_until

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        motion_mask = get_motion_mask(gray)
        faces = detect_faces(gray)

        h, w = image.shape[:2]
        left_faces = [tuple(face) for face in faces if face[0] + face[2] // 2 < w // 2]
        right_faces = [tuple(face) for face in faces if face[0] + face[2] // 2 >= w // 2]

        now = time.time()
        smiling = any(face_is_smiling(gray, tuple(face)) for face in faces)
        shielding = detect_shielding_motion(motion_mask, [tuple(face) for face in faces])

        state = "Neutral"
        if shielding or now < storm_manual_until:
            state = "Storm"
            image = apply_storm(image)
        elif smiling or now < sun_manual_until:
            state = "Sun"
            if sun_img is not None:
                image = overlay_transparent(image, cv2.resize(sun_img, (w, h)), 0, 0)

        for face in left_faces:
            image = draw_player(image, face, "left", left_character)
        for face in right_faces:
            image = draw_player(image, face, "right", right_character)

        image = draw_guns(image, left_faces, right_faces)
        image = process_body_gestures(image, motion_mask, left_faces, right_faces)
        image = process_grenades(image, left_faces, right_faces)
        image = process_bullets(image, left_faces, right_faces)
        image = process_explosions(image)
        image = draw_gesture_messages(image)
        image = draw_ui(image, state, left_faces, right_faces)

        cv2.imshow("Super Wings AR Game", image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            left_character = switch_character(left_character)
        elif key in (13, 10):
            right_character = switch_character(right_character)
        elif key == ord(" "):
            left_character, right_character = right_character, left_character
        elif key == ord("z"):
            shoot_from_faces("left", left_faces)
        elif key == ord("m"):
            shoot_from_faces("right", right_faces)
        elif key == ord("x"):
            throw_grenade_from_faces("left", left_faces)
        elif key == ord("n"):
            throw_grenade_from_faces("right", right_faces)
        elif key == ord("r"):
            storm_manual_until = time.time() + 4
        elif key == ord("s"):
            sun_manual_until = time.time() + 3

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
