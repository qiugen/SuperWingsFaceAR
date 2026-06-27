# 超级飞侠 AR 互动游戏 (Super Wings AR Game)

这是一个基于 Python、OpenCV 和 MediaPipe 制作的 Mac 摄像头 AR 互动游戏。

## 游戏功能
1. **超级飞侠换脸**：自动识别面部，并为您戴上“乐迪 (Jett)”的经典红色头盔和护目镜。
2. **表情互动特效**：
   - 😊 **当你笑的时候**：屏幕上会出现明亮的“太阳”特效。
   - 😢 **当你撇嘴/哭的时候**：屏幕上会下起“雨”特效（带动画效果）。

## 环境要求
- macOS (需授予终端/IDE 摄像头访问权限)
- Python 3
- 依赖库：`opencv-python`, `mediapipe`, `numpy`

## 如何运行

1. 进入项目目录：
```bash
cd /Users/bytedance/Project/SuperWingsFaceAR
```

2. 运行游戏主程序：
```bash
python3 ar_game.py
```

3. **退出游戏**：在弹出的摄像头窗口激活时，按下键盘上的 `Q` 键即可退出。

## 目录结构
- `ar_game.py`: 游戏主逻辑代码，负责摄像头捕获、面部关键点检测、表情识别和图像叠加。
- `create_assets.py`: 用于生成占位符素材（头盔、雨水、太阳）的脚本（已运行完毕）。
- `assets/`: 存放生成的透明 PNG 图像素材。
- `requirements.txt`: Python 依赖列表。