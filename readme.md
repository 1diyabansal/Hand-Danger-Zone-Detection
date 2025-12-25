Hand Danger Zone Detection Using OpenCV
✨ Overview

This project is a real-time computer vision system that detects a user’s hand from a webcam feed, identifies the fingertip, and checks how close it is to a predefined danger zone on the screen.
The system classifies the state into:

🟢 SAFE

🟠 WARNING

🔴 DANGER (flashing alert)

The project uses OpenCV, classical hand segmentation, contour analysis, and convex hull geometry.

🚀 Features

Real-time hand tracking

Accurate fingertip detection

Skin detection using HSV + YCrCb color spaces

Morphological filtering for clean masks

Distance-based safety classification

Visual overlays, FPS display, and flashing warnings

Fully customizable danger zone

🛠️ Tech Stack

Python

OpenCV

NumPy

📌 How It Works
1. Capture & preprocess frames

Resize, flip, and prepare webcam feed.

2. Skin Mask Generation

Combines HSV + YCrCb masks for robust skin detection.

3. Hand Contour Detection

Largest contour is treated as the hand.

4. Fingertip Estimation

Compute centroid using moments

Compute convex hull

Fingertip = hull point farthest from centroid

5. Zone Danger Classification

Distance from fingertip → predefined box.

6. Visual UI

Green for SAFE

Orange for WARNING

Red (with flashing overlay) for DANGER

Shows mask, fingertip, and centroid

📷 Example Output

Live hand mask

Bounding box danger zone

Fingertip marker

Flashing DANGER alert

▶️ Running the Project
python hand boundary.py


Press ESC to exit.
Press g to print current zone settings.

⚙️ Adjustable Parameters

Inside the code:

SAFE_DIST = 200
WARNING_DIST = 80
DANGER_DIST = 40


Danger box:

BOX = (x1, y1, x2, y2)


Tune these depending on resolution.

🧠 Future Work

Replace skin detection with ML segmentation

Add gesture recognition

Integrate into touchless PC control system

Add audio warnings

Add GUI for changing danger zone