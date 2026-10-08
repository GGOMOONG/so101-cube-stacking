#!/usr/bin/env python3
"""실물 놓기 직전 프레임(DAgger 3 보정 데이터, 쥔 구간 끝 8프레임 전)과 같은 관절 상태·빨강 위치로 만든 시뮬 렌더를 나란히 비교.
출력: calib/sim/place_compare_dagger3.png (행마다 실물 탑 | 시뮬 탑 | 실물 손목 | 시뮬 손목)"""
import sys, os, glob, numpy as np, pandas as pd, cv2, mujoco
os.chdir(os.path.expanduser("~/robotics")); sys.path.insert(0, ".")
import sim_phys as sp, cube_locate as cl
from sim_cam import distort, top_cam_xml
import sim_demo
DS = os.path.expanduser("~/.cache/huggingface/lerobot/moonbeards/dagger_stack3_corrections3_h264/")
df = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(DS+"data/*/*.parquet"))])
ep = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(DS+"meta/episodes/*/*.parquet"))]).set_index("episode_index")
def frame(cam, e, fi):
    r = ep.loc[e]; path = DS + f"videos/observation.images.{cam}/chunk-{int(r[f'videos/observation.images.{cam}/chunk_index']):03d}/file-{int(r[f'videos/observation.images.{cam}/file_index']):03d}.mp4"
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(r[f'videos/observation.images.{cam}/from_timestamp']*fps)) + fi); ok, im = cap.read(); return im
sp.PhysSim.CUBE_CONTACT = sp.PhysSim.STIFF_CUBE
rows = []; stats = []
EPS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(0, 20, 4))
for e in EPS:
    d = df[df.episode_index == e]; A = np.stack(d["action"].values); St = np.stack(d["observation.state"].values)
    held = A[:,5] < 5
    if not held.any(): print(e, "쥔 구간 없음"); continue
    last = np.where(held)[0][-1]; fi = max(0, last - 8); st = St[fi]
    top = frame("top", e, fi); wr = frame("wrist", e, fi)
    # 빨강 위치: 놓기 직전 프레임에서는 주황·그리퍼에 가려 빨강 윗면이 안 보이므로, 에피소드 마지막 프레임(쌓인 뒤)의 주황 윗면(높이 60 mm)으로 탑 위치를 잡는다
    topL = frame("top", e, len(d) - 1); locL, _ = cl.locate(topL); oL = locL.get("orange")
    if oL is None: print(e, "마지막 프레임 주황 못 찾음"); continue
    rx, ry = cl.pixel_to_plane(oL["px"][0], oL["px"][1], 60.0)
    K = float(os.environ.get("HIDDEN_K", "0")); comp = {"shoulder_lift": K, "elbow_flex": K * 0.5} if K > 0 else None
    s = sp.PhysSim({"red": (rx, ry, 0), "orange": (rx, ry + 200, 0)}, extra_xml=top_cam_xml(sp.Kin()), gripper_extra_xml=sim_demo.wrist_xml(), compliance=comp)
    for nm in ("pad_fixed", "pad_moving"): s.m.geom(nm).rgba[3] = 0.0
    s.set_state(st); s.set_ctrl(st)
    tip = s.tip_grid(s.extra(st[5])); p = np.array([tip[0], tip[1], tip[2] + 12.5])
    adr = s.m.joint("orange_free").qposadr[0]; s.d.qpos[adr:adr+3] = s.k.grid_to_base(p) / 1000.0
    # 큐브를 손목 roll 방향에 맞춰 돌림: 그리퍼 링크의 자세에서 z축 회전만 취함
    gl = s.m.body("gripper_link").id; mujoco.mj_forward(s.m, s.d); Rg = s.d.xmat[gl].reshape(3, 3)
    yaw = np.arctan2(Rg[1, 0], Rg[0, 0]); q = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]); s.d.qpos[adr+3:adr+7] = q
    mujoco.mj_forward(s.m, s.d)
    for _ in range(30): s.tick(st)
    tip = s.tip_grid(s.extra(st[5]))
    stop = distort(s.render_cam("top")); swr = distort(s.render_cam("wrist"))
    o, tilt = s.cube_grid("orange"); g = s.readings()[5]
    print(f"ep {e} 프레임 {fi} 빨강 ({rx:.0f},{ry:.0f}) 손끝 {np.round(tip,0)} 시뮬 주황 {np.round(o,0)} 그리퍼 실물 {st[5]:.1f} 시뮬 {g:.1f}")
    # 손목 화면에서 빨강 위치 비교(HSV)
    def red_center(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, (0,120,60), (8,255,255)) | cv2.inRange(hsv, (170,120,60), (180,255,255))
        ys, xs = np.nonzero(m); return (int(xs.mean()), int(ys.mean()), int(m.sum()//255)) if len(xs) > 200 else None
    rr, sr = red_center(wr), red_center(cv2.cvtColor(swr, cv2.COLOR_RGB2BGR)); print("   손목 화면 빨강 중심·면적: 실물", rr, "시뮬", sr); stats.append((e, rr, sr))
    row = np.hstack([top, cv2.cvtColor(stop, cv2.COLOR_RGB2BGR), wr, cv2.cvtColor(swr, cv2.COLOR_RGB2BGR)])
    cv2.putText(row, f"ep{e} real top | sim top | real wrist | sim wrist", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    rows.append(cv2.resize(row, (row.shape[1]//2, row.shape[0]//2)))
img = np.vstack(rows); os.makedirs("calib/sim", exist_ok=True); cv2.imwrite("calib/sim/place_compare_dagger3.png", img); print("저장 calib/sim/place_compare_dagger3.png", img.shape)
