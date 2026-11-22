import numpy as np
import cv2


def draw_aura(frame, pts, color, grow=35, blur=40, strength=0.65, scale=8):
    h, w = frame.shape[:2]
    small_pts = (np.array(pts, np.float32) / scale).astype(np.int32)
    mask = np.zeros((h // scale, w // scale), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(small_pts), 255)
    small_grow = max(1, grow // scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * small_grow + 1, 2 * small_grow + 1))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), blur / scale)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = (mask.astype(np.float32) * (strength / 255.0))[:, :, None]
    tint = np.array(color, np.float32)
    frame[:] = (frame * (1 - alpha) + tint * alpha).astype(np.uint8)


def draw_bolt(frame, start, angle, length, thickness, segments=6, jitter=12, color=(0, 0, 0)):
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    points = [np.array(start, float)]
    for i in range(1, segments + 1):
        step = np.array(start, float) + direction * length * i / segments
        points.append(step + normal * np.random.uniform(-jitter, jitter))
    cv2.polylines(frame, [np.array(points, np.int32)], False, color, thickness, cv2.LINE_AA)
    return points


def draw_hand_lightning(frame, pts, n_bolts=7, color=(0, 0, 0)):
    pts = np.array(pts, float)
    center = pts.mean(axis=0)
    hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2).astype(float)
    hull_dirs = hull - center
    hull_dirs /= np.maximum(np.linalg.norm(hull_dirs, axis=1, keepdims=True), 1e-6)
    offset = np.random.uniform(0, 2 * np.pi)
    for k in range(n_bolts):
        angle = offset + 2 * np.pi * k / n_bolts + np.random.uniform(-0.3, 0.3)
        target = np.array([np.cos(angle), np.sin(angle)])
        start = hull[np.argmax(hull_dirs @ target)]
        length = np.random.uniform(60, 120)
        points = draw_bolt(frame, start, angle, length, 4, color=color)
        if np.random.rand() < 0.6:
            fork_start = points[len(points) // 2]
            fork_angle = angle + np.random.choice([-1, 1]) * np.random.uniform(0.4, 0.8)
            draw_bolt(frame, fork_start, fork_angle, length * 0.5, 2, segments=4, jitter=8, color=color)
