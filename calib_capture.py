#!/usr/bin/env python3
"""체스판 촬영 도구. 카메라 창을 띄우고 스페이스로 저장, q로 종료.
사용: python ~/robotics/calib_capture.py 0            (카메라 인덱스 0, 기본 저장 폴더 ~/robotics/calib/chess)
      python ~/robotics/calib_capture.py 0 ~/robotics/calib/grid   (격자 촬영 때는 폴더만 바꿈)
체스판 안쪽 모서리(6×9)가 잡히면 화면에 색선이 그려지고, 그때 스페이스를 누르면 저장된다.
"""
import sys, os, time
import cv2

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/robotics/calib/chess")
os.makedirs(out, exist_ok=True)
PATTERN = (6, 9)  # 안쪽 모서리 (가로 6, 세로 9) — 7×10칸 체스판

cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
n = len([f for f in os.listdir(out) if f.endswith(".png")])
print(f"카메라 {idx}, 저장 폴더 {out}, 기존 {n}장. 스페이스=저장, q=종료", flush=True)
while True:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.05)
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
    view = frame.copy()
    if found:
        cv2.drawChessboardCorners(view, PATTERN, corners, found)
    cv2.putText(view, f"saved {n}  chessboard {'OK' if found else '--'}  [space]=save [q]=quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if found else (0, 0, 255), 2)
    cv2.imshow("calib capture", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        n += 1
        path = os.path.join(out, f"img_{n:02d}.png")
        cv2.imwrite(path, frame)
        print("saved", path, "(chessboard found)" if found else "(chessboard NOT found)", flush=True)
    elif k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
print("done, total", n)
