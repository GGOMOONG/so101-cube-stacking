#!/usr/bin/env python3
"""보정용 출력물 PDF 생성 (A4 2장).
  1쪽: 체스판 7×10칸(안쪽 모서리 6×9), 한 칸 25 mm. 카메라 내부 파라미터 보정용.
  2쪽: 테이블 좌표 격자 20 mm + ArUco 마커 4개(4×4_50, id 0~3, 40 mm). 탑 카메라 → 테이블 평면 호모그래피용.
반드시 "실제 크기(100%)"로 인쇄하고, 각 쪽의 100 mm 눈금을 자로 확인한다.
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

OUT = "/Users/m4pro/robotics/calibration_targets.pdf"
kr = fm.FontProperties(fname="/System/Library/Fonts/Supplemental/AppleGothic.ttf")
A4_W, A4_H = 210.0, 297.0  # mm
IN = 25.4


def new_page():
    fig = plt.figure(figsize=(A4_W / IN, A4_H / IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W)
    ax.set_ylim(0, A4_H)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def scale_bar(ax, x, y, label):
    ax.plot([x, x + 100], [y, y], color="k", lw=1.2)
    for k in range(0, 101, 10):
        ax.plot([x + k, x + k], [y, y + (3 if k % 50 == 0 else 1.5)], color="k", lw=0.8)
    ax.text(x, y - 4, label, fontproperties=kr, fontsize=8, va="top")


with PdfPages(OUT) as pdf:
    # ---------- page 1: checkerboard ----------
    fig, ax = new_page()
    sq, cols, rows = 25.0, 7, 10          # inner corners: 6 x 9
    bw, bh = cols * sq, rows * sq         # 175 x 250 mm
    x0, y0 = (A4_W - bw) / 2, (A4_H - bh) / 2 - 4
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((x0 + c * sq, y0 + r * sq), sq, sq, color="k", lw=0))
    ax.add_patch(Rectangle((x0, y0), bw, bh, fill=False, lw=0.5, color="k"))
    ax.text(A4_W / 2, y0 + bh + 12, "1쪽 · 체스판  7×10칸, 한 칸 25 mm  (OpenCV 안쪽 모서리 6×9)",
            fontproperties=kr, fontsize=11, ha="center")
    ax.text(A4_W / 2, y0 + bh + 6, "용도: 카메라 내부 파라미터(초점거리·왜곡) 보정. 딱딱한 판에 붙여 여러 각도로 15~20장 촬영.",
            fontproperties=kr, fontsize=8, ha="center")
    scale_bar(ax, x0, y0 - 8, "100 mm  ← 자로 확인 (실제 크기 100% 인쇄)")
    ax.text(x0 + bw, y0 - 8, "SO-101 보정 · 2026-09", fontproperties=kr, fontsize=7, ha="right", va="top")
    pdf.savefig(fig); fig.savefig(OUT.replace(".pdf","_p1.pdf"))
    plt.close(fig)

    # ---------- page 2: table grid + ArUco ----------
    fig, ax = new_page()
    step = 20.0
    gx0, gy0 = 15.0, 25.0
    gw, gh = 180.0, 240.0                  # 9 x 12 cells
    # grid
    for k in range(int(gw / step) + 1):
        x = gx0 + k * step
        ax.plot([x, x], [gy0, gy0 + gh], color="0.25", lw=0.5 if k % 5 else 1.0)
        ax.text(x, gy0 - 3, f"{int(k*step)}", fontsize=6, ha="center", va="top")
    for k in range(int(gh / step) + 1):
        y = gy0 + k * step
        ax.plot([gx0, gx0 + gw], [y, y], color="0.25", lw=0.5 if k % 5 else 1.0)
        ax.text(gx0 - 2, y, f"{int(k*step)}", fontsize=6, ha="right", va="center")
    # cross marks at every grid node (easy to click in an image)
    for i in range(int(gw / step) + 1):
        for j in range(int(gh / step) + 1):
            x, y = gx0 + i * step, gy0 + j * step
            ax.plot([x - 1.5, x + 1.5], [y, y], color="k", lw=0.8)
            ax.plot([x, x], [y - 1.5, y + 1.5], color="k", lw=0.8)
    # origin
    ax.plot(gx0, gy0, "o", color="k", ms=6)
    ax.text(gx0 + 3, gy0 + 3, "원점 (0,0)  x→  y↑  단위 mm", fontproperties=kr, fontsize=8)
    # ArUco markers (DICT_4X4_50, ids 0-3), 40 mm, placed on grid nodes
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    msize = 40.0
    spots = {0: (gx0 + 20, gy0 + 20), 1: (gx0 + gw - 20 - msize, gy0 + 20),
             2: (gx0 + 20, gy0 + gh - 20 - msize), 3: (gx0 + gw - 20 - msize, gy0 + gh - 20 - msize)}
    for mid, (mx, my) in spots.items():
        img = cv2.aruco.generateImageMarker(d, mid, 300, borderBits=1)
        ax.add_patch(Rectangle((mx - 4, my - 4), msize + 8, msize + 8, color="w", lw=0))  # white quiet zone
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(mx, mx + msize, my, my + msize), interpolation="nearest", zorder=3)
        ax.text(mx + msize / 2, my - 5.5, f"id {mid}  좌하 모서리 ({int(mx-gx0)}, {int(my-gy0)})",
                fontproperties=kr, fontsize=6, ha="center", va="top", zorder=4)
    ax.text(A4_W / 2, gy0 + gh + 10, "2쪽 · 테이블 좌표 격자 20 mm + ArUco 마커(4×4_50, id 0~3, 40 mm)",
            fontproperties=kr, fontsize=11, ha="center")
    ax.text(A4_W / 2, gy0 + gh + 4, "용도: 탑 카메라 픽셀 → 테이블 mm 변환(호모그래피). 로봇 앞 테이블에 테이프로 고정.",
            fontproperties=kr, fontsize=8, ha="center")
    scale_bar(ax, gx0 + 60, gy0 - 12, "100 mm  ← 자로 확인 (실제 크기 100% 인쇄)")
    pdf.savefig(fig); fig.savefig(OUT.replace(".pdf","_p2.pdf"))
    plt.close(fig)

print("saved", OUT)
