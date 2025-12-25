

import cv2
import numpy as np
import time


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BLUR = 7
MORPH_KERNEL = (7,7)
HSV_LOWER = np.array([0, 30, 60])
HSV_UPPER = np.array([50, 200, 255])
YCRCB_LOWER = np.array([0, 135, 85])
YCRCB_UPPER = np.array([255, 180, 135])
BOX = (int(FRAME_WIDTH*0.7), int(FRAME_HEIGHT*0.2), FRAME_WIDTH-20, int(FRAME_HEIGHT*0.8))
SAFE_DIST = 200
WARNING_DIST = 80
DANGER_DIST = 40
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)

def skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask_hsv = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask_ycrcb = cv2.inRange(ycrcb, YCRCB_LOWER, YCRCB_UPPER)
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def find_hand_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
       return None, 0
    max_c = max(contours, key=cv2.contourArea)
    return max_c, cv2.contourArea(max_c)


def fingertip_from_contour(cnt):
    if cnt is None or len(cnt) < 5:
       return None
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    centroid = np.array([cx, cy])
    hull = cv2.convexHull(cnt, returnPoints=True)
    hull_pts = hull.reshape(-1, 2)
    dists = np.linalg.norm(hull_pts - centroid, axis=1)
    idx = np.argmax(dists)
    fingertip = tuple(hull_pts[idx])
    return fingertip, (cx, cy)

def point_to_rect_distance(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect

    if x1 <= x <= x2 and y1 <= y <= y2:
      return 0
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return int(np.hypot(dx, dy))

def classify_state(dist):
    if dist == 0 or dist <= DANGER_DIST:
       return "DANGER"
    elif dist <= WARNING_DIST:
       return "WARNING"
    elif dist <= SAFE_DIST:
        return "WARNING"
    else:
        return "SAFE"


def draw_ui(frame, state, fingertip, centroid, rect, fps):
    x1, y1, x2, y2 = rect
    color_map = {"SAFE": (0,255,0), "WARNING": (0,165,255), "DANGER": (0,0,255)}
    rect_color = color_map.get(state, (255,255,255))
    cv2.rectangle(frame, (x1,y1), (x2,y2), rect_color, 2)
    cv2.putText(frame, f"STATE: {state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rect_color, 2)
    if fingertip is not None:
       cv2.circle(frame, fingertip, 8, (255,255,255), -1)
       cv2.putText(frame, "Fingertip", (fingertip[0]+10, fingertip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    if centroid is not None:
       cv2.circle(frame, centroid, 5, (200,200,200), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    if state == "DANGER":
       if int(time.time()*2) % 2 == 0:
          h, w = frame.shape[:2]
          overlay = frame.copy()
          cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
          alpha = 0.25
          cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
          cv2.putText(frame, "DANGER DANGER", (int(w*0.15), int(h*0.5)), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255,255,255), 6)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    prev_time = time.time()
    fps = 0.0
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
           print("Can't read camera. Exiting.")
           break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        mask = skin_mask(frame)
        cnt, area = find_hand_contour(mask)
        fingertip = None
        centroid = None
        if cnt is not None and area > 2000:
           fp = fingertip_from_contour(cnt)
           if fp is not None:
              fingertip, centroid = fp
           cv2.drawContours(frame, [cnt], -1, (150,150,150), 1)
        if fingertip is not None:
           dist = point_to_rect_distance(fingertip, BOX)
        elif centroid is not None:
            dist = point_to_rect_distance(centroid, BOX)
        else:
            dist = 9999
        state = classify_state(dist)
        frame = draw_ui(frame, state, fingertip, centroid, BOX, fps)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        t1 = time.time()
        dt = t1 - t0
        if dt > 0:
            fps = 0.9*fps + 0.1*(1.0/dt)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
           break
        if key == ord('g'):
           print("BOX:", BOX, "SAFE_DIST:", SAFE_DIST, "WARNING_DIST:", WARNING_DIST, "DANGER_DIST:", DANGER_DIST)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()