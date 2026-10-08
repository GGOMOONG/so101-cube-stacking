#!/usr/bin/env python3
"""실물 놓기 직전 손목 프레임(DAgger 3 보정)으로 손목 카메라 자세 잔차를 맞춘다.
hand-eye 결과(calib/wrist_cam.json mjcf)에 카메라 자체 좌표계의 (pitch, 앞뒤 dz, 상하 dy) 섭동을 주고,
손목 화면에서 주황(손안)·빨강(책상, 쌓인 뒤 탑 카메라로 위치 확정)의 중심·면적이 실물과 맞는 조합을 격자 탐색.
사용: python wrist_fit_release.py [--apply]    → 최적 섭동 출력, --apply면 wrist_cam.json의 mjcf를 갱신(원본은 mjcf_handeye로 보존)"""
import sys, os, glob, json, re, numpy as np, pandas as pd, cv2, mujoco
os.chdir(os.path.expanduser("~/robotics")); sys.path.insert(0, ".")
import sim_phys as sp, cube_locate as cl, sim_demo
from sim_cam import distort, top_cam_xml
DS = os.path.expanduser("~/.cache/huggingface/lerobot/moonbeards/dagger_stack3_corrections3_h264/")
df = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(DS+"data/*/*.parquet"))])
ep = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(DS+"meta/episodes/*/*.parquet"))]).set_index("episode_index")
def frame(cam, e, fi):
    r = ep.loc[e]; path = DS + f"videos/observation.images.{cam}/chunk-{int(r[f'videos/observation.images.{cam}/chunk_index']):03d}/file-{int(r[f'videos/observation.images.{cam}/file_index']):03d}.mp4"
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(r[f'videos/observation.images.{cam}/from_timestamp']*fps)) + fi); ok, im = cap.read(); return im
def blob(bgr, color):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, (0,120,60), (8,255,255)) | cv2.inRange(hsv, (170,120,60), (180,255,255))) if color == "red" else cv2.inRange(hsv, (9,120,80), (22,255,255))
    ys, xs = np.nonzero(m); return np.array([xs.mean(), ys.mean(), len(xs)], float) if len(xs) > 150 else None

# 실물 프레임 수집
samples = []
for e in range(20):
    d = df[df.episode_index == e]; A = np.stack(d["action"].values); St = np.stack(d["observation.state"].values)
    held = np.where(A[:,5] < 5)[0]
    if len(held) == 0: continue
    fi = max(0, held[-1] - 8); st = St[fi]
    topL = frame("top", e, len(d) - 1); locL, _ = cl.locate(topL); oL = locL.get("orange")
    if oL is None: continue
    rx, ry = cl.pixel_to_plane(oL["px"][0], oL["px"][1], 60.0)
    wr = frame("wrist", e, fi); bo, br = blob(wr, "orange"), blob(wr, "red")
    if bo is None or br is None: continue
    samples.append(dict(e=e, st=st, red=(rx, ry), bo=bo, br=br))
print(f"실물 표본 {len(samples)}개")

wj = json.load(open("calib/wrist_cam.json")); base = wj["mjcf"] if os.environ.get("FROM_CURRENT") else wj.get("mjcf_handeye", wj["mjcf"])
pos0 = np.array(re.search(r'pos="([^"]+)"', base).group(1).split(), float); q0 = np.array(re.search(r'quat="([^"]+)"', base).group(1).split(), float)
def cam_xml(pitch_deg, dy_mm, dz_mm):
    R = np.empty(9); mujoco.mju_quat2Mat(R, q0); R = R.reshape(3, 3)
    pos = pos0 + R @ np.array([0, dy_mm, dz_mm]) / 1000.0
    dq = np.array([np.cos(np.deg2rad(pitch_deg) / 2), np.sin(np.deg2rad(pitch_deg) / 2), 0, 0]); q = np.empty(4); mujoco.mju_mulQuat(q, q0, dq)
    x = re.sub(r'pos="[^"]+"', f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"', base, count=1)
    return re.sub(r'quat="[^"]+"', f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"', x, count=1)

sp.PhysSim.CUBE_CONTACT = sp.PhysSim.STIFF_CUBE
def evaluate(pitch, dy, dz, verbose=False):
    s = sp.PhysSim({"red": (0, 150, 0), "orange": (100, 150, 0)}, extra_xml=top_cam_xml(sp.Kin()), gripper_extra_xml=cam_xml(pitch, dy, dz))
    for nm in ("pad_fixed", "pad_moving"): s.m.geom(nm).rgba[3] = 0.0
    ar, ao = s.m.joint("red_free").qposadr[0], s.m.joint("orange_free").qposadr[0]
    cost = 0.0; n = 0
    for smp in samples:
        st = smp["st"]; s.set_state(st); s.set_ctrl(st)
        s.d.qpos[ar:ar+3] = s.k.grid_to_base((smp["red"][0], smp["red"][1], 15.0)) / 1000.0; s.d.qpos[ar+3:ar+7] = [1, 0, 0, 0]
        mujoco.mj_forward(s.m, s.d)
        tip = s.tip_grid(s.extra(st[5])); s.d.qpos[ao:ao+3] = s.k.grid_to_base((tip[0], tip[1], tip[2] + 12.5)) / 1000.0
        gl = s.m.body("gripper_link").id; Rg = s.d.xmat[gl].reshape(3, 3); yaw = np.arctan2(Rg[1, 0], Rg[0, 0]); s.d.qpos[ao+3:ao+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        mujoco.mj_forward(s.m, s.d)
        im = cv2.cvtColor(distort(s.render_cam("wrist")), cv2.COLOR_RGB2BGR); bo, br = blob(im, "orange"), blob(im, "red")
        for b, t in ((bo, smp["bo"]), (br, smp["br"])):
            if b is None: cost += 4.0; continue
            cost += ((b[0] - t[0]) / 100) ** 2 + ((b[1] - t[1]) / 100) ** 2 + (np.log(b[2] / t[2])) ** 2
        n += 1
        if verbose: print(f"  ep {smp['e']}: 주황 실물 {np.round(smp['bo'])} 시뮬 {None if bo is None else np.round(bo)} | 빨강 실물 {np.round(smp['br'])} 시뮬 {None if br is None else np.round(br)}")
    return cost / max(n, 1)

if __name__ == "__main__":
    print("현재(hand-eye+3°):"); c0 = evaluate(0, 0, 0, verbose=True); print("  비용", round(c0, 3))
    best = (c0, (0, 0, 0))
    PO = bool(os.environ.get("PITCH_ONLY"))
    for pitch in (-20, -15, -10, -7, -5, -3, 0, 3, 5, 7, 10, 15, 20):
        for dy in ((0,) if PO else (-20, -10, 0, 10, 20)):
            for dz in ((0,) if PO else (-20, -10, 0, 10, 20)):
                c = evaluate(pitch, dy, dz)
                if c < best[0]: best = (c, (pitch, dy, dz)); print("  개선", round(c, 3), (pitch, dy, dz), flush=True)
    print("최적:", best); evaluate(*best[1], verbose=True)
    if "--apply" in sys.argv and best[1] != (0, 0, 0):
        wj["mjcf_handeye"] = base; wj["mjcf"] = cam_xml(*best[1]); wj["release_fit"] = dict(pitch_deg=best[1][0], dy_mm=best[1][1], dz_mm=best[1][2], cost=best[0], cost_before=c0, n=len(samples))
        json.dump(wj, open("calib/wrist_cam.json", "w"), indent=1); print("wrist_cam.json 갱신")
