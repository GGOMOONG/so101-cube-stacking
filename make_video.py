#!/usr/bin/env python3
"""2분 요약 영상 조립 (실제 녹화본만 사용, 생성 없음). 재료: 평가 데이터셋 mp4, 휴대폰 영상, 사진, 도식.
사용: python ~/robotics/make_video.py  → videos/summary_2min_draft.mp4 (1280x720, 30 fps, 무음)
장면 정의는 아래 SCENES. 각 항목: ("종류", 인자..., 자막 한/영). 종류: clip(데이터셋, 에피소드, 카메라, 배속, 최대초) / file(경로, 시작초, 끝초, 배속) / image(경로, 초) / slides([경로...], 장당 초) / title(배경경로, 초, 큰제목, 부제)
"""
import os, glob, cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
W, H, FPS = 1280, 720, 30
HF = os.path.expanduser("~/.cache/huggingface/lerobot/moonbeards")
R = os.path.expanduser("~/robotics")
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
f_ko = ImageFont.truetype(FONT, 34); f_en = ImageFont.truetype(FONT, 26); f_title = ImageFont.truetype(FONT, 64); f_sub = ImageFont.truetype(FONT, 36); f_small = ImageFont.truetype(FONT, 22)


def fit(img, bottom_pad=0):
    """BGR 이미지를 1280x720 안에 비율 유지로 맞추고 검정으로 채움. bottom_pad만큼 아래를 비워 자막과 안 겹치게."""
    h, w = img.shape[:2]; Hh = H - bottom_pad; s = min(W / w, Hh / h); nw, nh = int(w * s), int(h * s)
    out = np.zeros((H, W, 3), np.uint8); r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (Hh - nh) // 2, (W - nw) // 2; out[y:y + nh, x:x + nw] = r; return out


def caption(frame, ko, en=None, tag=None):
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA"); ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    bar_h = 110 if en else 70; d.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, 150))
    d.text((40, H - bar_h + 14), ko, font=f_ko, fill=(255, 255, 255, 255))
    if en: d.text((40, H - bar_h + 62), en, font=f_en, fill=(220, 220, 220, 255))
    if tag: d.rectangle([W - 300, 20, W - 20, 62], fill=(0, 0, 0, 140)); d.text((W - 290, 26), tag, font=f_small, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(Image.alpha_composite(im, ov).convert("RGB")), cv2.COLOR_RGB2BGR)


def title_card(bg_path, secs, title, sub):
    bg = fit(cv2.imread(bg_path)); im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 120)); d = ImageDraw.Draw(ov)
    d.text((60, 250), title, font=f_title, fill=(255, 255, 255, 255)); d.text((60, 350), sub, font=f_sub, fill=(230, 230, 230, 255))
    fr = cv2.cvtColor(np.array(Image.alpha_composite(im, ov).convert("RGB")), cv2.COLOR_RGB2BGR)
    return [fr] * int(secs * FPS)


def ep_range(ds, ep, cam="top"):
    e = pd.read_parquet(glob.glob(f"{HF}/{ds}/meta/episodes/**/*.parquet", recursive=True)[0]); row = e[e.episode_index == ep].iloc[0]
    return float(row[f"videos/observation.images.{cam}/from_timestamp"]), float(row[f"videos/observation.images.{cam}/to_timestamp"])


def read_clip(path, t0, t1, speed, max_s=None):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30; cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    out = []; n_src = 0; step = fps * speed / FPS; nxt = 0.0; limit = (t1 - t0) if max_s is None else min(t1 - t0, max_s * speed)
    while True:
        ok, f = cap.read()
        if not ok or n_src / fps >= limit: break
        if n_src >= nxt: out.append(fit(f)); nxt += step
        n_src += 1
    cap.release(); return out


def clip(ds, ep, cam, speed, max_s):
    t0, t1 = ep_range(ds, ep, cam); return read_clip(f"{HF}/{ds}/videos/observation.images.{cam}/chunk-000/file-000.mp4", t0, t1, speed, max_s)


def build(scenes, out_path):
    frames_total = 0; writer = None
    for sc in scenes:
        kind = sc[0]; ko, en = sc[-2], sc[-1]; tag = None
        if kind == "clip": frs = clip(*sc[1:6]); tag = sc[6] if len(sc) > 8 else None
        elif kind == "file": frs = read_clip(sc[1], sc[2], sc[3], sc[4]); tag = sc[5] if len(sc) > 7 else None
        elif kind == "image": frs = [fit(cv2.imread(sc[1]), 110)] * int(sc[2] * FPS)
        elif kind == "slides": frs = [fr for p in sc[1] for fr in [fit(cv2.imread(p), 110)] * int(sc[2] * FPS)]
        elif kind == "title": frs = title_card(sc[1], sc[2], sc[3], sc[4]); ko = None
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
            if not writer.isOpened(): writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for fr in frs: writer.write(caption(fr, ko, en, tag) if ko else fr)
        frames_total += len(frs); print(f"  {kind:6s} {len(frs)/FPS:5.1f}s  누적 {frames_total/FPS:5.1f}s  {ko or sc[3]}")
    writer.release(); print("저장:", out_path, f"{frames_total/FPS:.1f}s")


V4 = "rollout_stack_unstack2_v4_stack_eval_v1_20260921_180005"; V1 = "rollout_color_cube_stack2_eval_v1_20260916_020659"
DAG = "rollout_dagger_stack_unstack2_stack2_20260921_015414"; CF = "rollout_stack_unstack2_v4_counterfactual_v1_20260921_182735"; DEMO = "color_cube_stack_unstack2"
steps = sorted(glob.glob(f"{R}/calib/pick_0929_042006_[1-8]_*_top.png")); wsteps = sorted(glob.glob(f"{R}/calib/pick_0929_042006_[1-8]_*_wrist.png")); sims = sorted(glob.glob(f"{R}/figures/sim/plan_*.png"))
SIMV = f"{R}/videos/sim_policy_vid_sim2_10k_A_3.mp4"        # 시뮬 v2 10k 성공 회차(시뮬 내, HUD 포함) — B 평가 뒤 녹화
REALV = f"{R}/videos/sim2_real_success.MP4"             # 시뮬 정책 실물 성공 장면(휴대폰 촬영, 있으면 사용)
SCENES = [
 ("title", f"{R}/photos/setup_full1.JPG", 6, "SO-101 색 큐브 쌓기", "모방학습 · 고전 파이프라인 · 시뮬→실물 / Learned · classical · sim-to-real"),
 ("clip", V4, 3, "top", 2.0, 9, "학습 정책 v4", "과제: 주황 큐브를 빨강 위에 쌓기. 성공 = 3초 유지", "Task: stack the orange cube on the red one; success = holds for 3 s"),
 ("clip", DEMO, 0, "top", 3.0, 4, "사람 시연", "사람 시연 176회 (왼손용 리더 암)", "176 human demonstrations (left-handed leader arm)"),
 ("clip", V1, 0, "top", 3.0, 4, "ACT v1", "첫 정책 ACT v1: 1/10. 시연 구조가 문제였다", "First policy, ACT v1: 1/10. The demonstration structure was the problem"),
 ("clip", DAG, 2, "top", 3.0, 4, "DAgger 교정", "DAgger: 정책이 멈추면 사람이 이어받아 교정 (40회)", "DAgger: a human takes over when the policy stalls (40 corrections)"),
 ("clip", V4, 4, "top", 2.5, 6, "학습 정책 v4", "SmolVLA + RTC + DAgger → v4: 원래 자리에서 쌓기 9/10", "SmolVLA + RTC + DAgger → v4: 9/10 in the original room"),
 ("file", f"{R}/videos/loop_stack_unstack1.MP4", 0, 45, 4.0, "언어 조건 루프", "한 모델, 두 문장: \"쌓아라\" / \"허물어라\"", "One model, two sentences: stack / unstack"),
 ("clip", CF, 0, "top", 3.0, 3, "반사실", "이미 쌓여 있으면 가만히 (5/5)", "Already stacked → no action (5/5)"),
 ("file", f"{R}/videos/loop_stack_unstack2.MP4", 20, 60, 4.0, "루프 실패", "시연 범위를 벗어나면 회복하지 못한다", "Outside the demonstration range it cannot recover"),
 ("image", f"{R}/calib/top_homography_check.png", 3, "고전 파이프라인 1/3: 체스판·격자로 카메라 보정 (0.45 mm)", "Classical 1/3: camera calibration from checkerboard and grid (0.45 mm)"),
 ("image", f"{R}/calib/cube_01_located.png", 3, "고전 2/3: 색 분할로 큐브 위치 mm 단위 (≤ 3 mm)", "Classical 2/3: cube position in mm from color segmentation (≤ 3 mm)"),
 ("slides", sims, 0.6, "고전 3/3: 관절 보정 → 역기구학 → MuJoCo로 궤적 검증", "Classical 3/3: joint calibration → inverse kinematics → MuJoCo check"),
 ("slides", steps, 1.0, "고전 파이프라인 실행: 8단계 집기·쌓기 (탑 카메라)", "Classical pipeline run: 8-step pick and stack (top camera)"),
 ("slides", wsteps, 0.9, "손목 카메라 시각 서보로 잡기 직전 0.4~1.5 mm 정렬 → 잡기 20/20", "Wrist-camera visual servo aligns to 0.4–1.5 mm → grasp 20/20"),
 ("image", f"{R}/figures/fig_results_ko.png", 9, "시연 범위 안 10/10 대 4/10 · 범위 밖 7/10 대 0/10", "In-distribution 10/10 vs 4/10 · out-of-distribution 7/10 vs 0/10"),
 ("image", f"{R}/figures/fig_pipelines_ko.png", 6, "두 방식은 다른 곳에서 무너진다 → 그러면 사람 시연 없이 시뮬레이터만으로는?", "They fail in different places → what if there were no human demonstrations at all?"),
 *([("file", SIMV, 0, 45, 1.0, "시뮬 전용 정책 (MuJoCo)", "MuJoCo: 실측 동역학·카메라, 자동 생성 시연 296회, 사람 시연 0회", "MuJoCo: measured dynamics, 296 auto-generated demos, zero human demos")] if os.path.exists(SIMV) else []),
 *([("file", REALV, 0, 40, 3.0, "시뮬 정책 → 실물", "같은 정책을 실물 팔에: 실물 시연 없이 쌓기 완수 (2/10)", "The same policy on the real arm: stacks without any real demonstration (2/10)")] if os.path.exists(REALV) else []),
 ("image", f"{R}/figures/fig_sim_curves_ko.png", 6, "시뮬 안 천장 약 4/10, 더 학습하면 오히려 떨어진다 (시연이 너무 균일)", "Ceiling ~4/10 even in sim; more training hurts (demos too uniform)"),
 ("image", f"{R}/figures/fig_sim2real_ko.png", 8, "실물 A: 시뮬 전용 2/10 대 실물 시연 3~4/10 · 시뮬 성적은 전이를 예측 못 함", "Real arm: sim-only 2/10 vs real-demo 3–4/10 · sim score ≠ transfer"),
 ("image", f"{R}/figures/fig_pipelines_ko.png", 5, "다음: 시뮬 시연 다양화, 놓기 서보, 혼합 제어", "Next: diversify simulated demos, closed-loop placing, hybrid control"),
 ("title", f"{R}/photos/setup_full2.JPG", 5, "moonbeards · SO-101 · 2026-09/10", "코드·연구노트·데이터: github (준비 중)"),
]
if __name__ == "__main__":
    os.makedirs(f"{R}/videos", exist_ok=True); build(SCENES, f"{R}/videos/summary_2min_draft.mp4")
