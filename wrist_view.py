#!/usr/bin/env python3
"""손목 카메라(인덱스 1) 큐브 측정: 주황 큐브의 화면 중심(u,v), 윗면 기울기(각도, 화면 x축 기준 −45~45°), 폭(px)."""
import cv2, numpy as np
WRIST_IDX = 1
ORANGE = [((4, 90, 90), (22, 255, 255))]


def open_wrist(idx=WRIST_IDX):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(5): cap.read()
    return cap


def grab(cap, n=5):
    f = None
    for _ in range(n):
        ok, fr = cap.read()
        if ok: f = fr
    return f


def measure_frame(f, color=ORANGE):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV); mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in color: mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 800]
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea); m = cv2.moments(c)
    (rx, ry), (rw, rh), ang = cv2.minAreaRect(c)
    ang = ((ang + 45) % 90) - 45
    return {"u": m["m10"] / m["m00"], "v": m["m01"] / m["m00"], "angle": float(ang), "w_px": float(min(rw, rh)), "area": float(cv2.contourArea(c)), "cnt": c}


def wrist_measure(cap, save=None):
    f = grab(cap)
    if f is None: return None
    r = measure_frame(f)
    if save is not None:
        vis = f.copy()
        if r is not None:
            cv2.drawContours(vis, [r["cnt"]], -1, (255, 255, 0), 2); cv2.circle(vis, (int(r["u"]), int(r["v"])), 5, (255, 255, 255), -1)
        cv2.imwrite(save, vis)
    return r
