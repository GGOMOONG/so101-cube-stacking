#!/usr/bin/env python3
"""MuJoCo 2단계: 가상 카메라. 탑 카메라는 실측 내부 파라미터(K, 왜곡)와 자세(호모그래피 rvec/tvec)로, 손목 카메라는 gripper_link에 붙여 실물 손목 사진의 턱 위치와 맞춰 추정.
사용:
  python ~/robotics/sim_cam.py --real calib/cam_check_top.png calib/cam_check_wrist.png   # 실물 사진의 큐브 위치를 읽어 같은 장면을 렌더, 나란히 저장(calib/sim/cam_compare_*.png)
  옵션: --wrist L H X (손목 카메라 위치: 턱 끝에서 뒤로 L mm, 턱 평면에서 H mm, 옆으로 X mm) --wrist-tilt 도 (턱 쪽으로 기울임) --grip 읽기값 --joints 6개
모듈: from sim_cam import cams_xml, distort;  PhysSim(cubes, extra_xml=top_xml, gripper_extra_xml=wrist_xml); img = distort(s.render_cam("top"))
"""
import os, sys, argparse
import numpy as np, cv2, mujoco
sys.path.insert(0, os.path.expanduser("~/robotics"))
from sim_phys import PhysSim, quat_of, _fmt, HOME
from gripper_geom import GripperGeom

CAL = os.path.expanduser("~/robotics/calib")
_c = np.load(os.path.join(CAL, "top_intrinsics.npz")); K, DIST = _c["K"], _c["dist"]
_h = np.load(os.path.join(CAL, "top_homography.npz")); R_CV, _ = cv2.Rodrigues(_h["rvec"]); T_CV = _h["tvec"].ravel()
W, H = 640, 480
WRIST_DEFAULT = dict(L=140.0, H=45.0, X=0.0, tilt=5.0, fovy=None)   # 손목 카메라 추정 초기값 (mm, 도). 렌즈는 탑과 같은 모듈로 가정


def top_cam_xml(k):
    """OpenCV 카메라(격자→카메라 R,t) → MuJoCo 카메라(베이스 좌표). MuJoCo는 −z를 보고 +y가 위, OpenCV는 +z를 보고 +y가 아래."""
    cam_grid = -R_CV.T @ T_CV                                    # 카메라 위치 (격자 mm)
    axes_grid = R_CV.T @ np.diag([1, -1, -1])                    # 열 = MuJoCo 카메라 x,y,z 축 (격자 좌표)
    pos = k.grid_to_base(cam_grid) / 1000.0; Rb = k.R @ axes_grid
    fovy = 2 * np.degrees(np.arctan(H / 2 / K[1, 1]))
    return (f'<camera name="top" pos="{_fmt(pos)}" quat="{_fmt(quat_of(Rb))}" '
            f'resolution="{W} {H}" focalpixel="{K[0,0]:.2f} {K[1,1]:.2f}" principalpixel="{W/2-K[0,2]:.2f} {H/2-K[1,2]:.2f}" sensorsize="0.0036 0.0027"/>')


def wrist_cam_xml(L=140.0, H_=35.0, X=0.0, tilt=0.0, fovy=None):
    """gripper_link 안의 손목 카메라. 턱 끝에서 뒤로 L, 턱 평면에서 H(회전축 방향), 옆으로 X(열림 방향) mm. 턱 끝 쪽을 보며 화면 위 = 턱 평면에서 멀어지는 쪽."""
    gg = GripperGeom(); lever = gg.lever / np.linalg.norm(gg.lever); axis = gg.axis / np.linalg.norm(gg.axis)
    n_open = np.cross(axis, lever); n_open /= np.linalg.norm(n_open)
    up = -axis                                                    # 부호는 실물 사진과 비교해 정함(턱이 화면 아래에 보여야 함)
    z = -lever                                                    # MuJoCo 카메라는 −z를 봄 → −z = lever (턱 끝 방향)
    # 기울임: 턱 평면 쪽으로 tilt°
    c, s_ = np.cos(np.deg2rad(tilt)), np.sin(np.deg2rad(tilt)); z = c * z + s_ * up; up = c * up - s_ * (-lever)
    x = np.cross(up, z); x /= np.linalg.norm(x); up = np.cross(z, x)
    Rc = np.column_stack([x, up, z])
    pos = gg.tip_gl - L / 1000.0 * lever + H_ / 1000.0 * (-up if False else up) + X / 1000.0 * n_open
    fv = fovy if fovy else 2 * np.degrees(np.arctan(H / 2 / K[1, 1]))
    return f'<camera name="wrist" pos="{_fmt(pos)}" quat="{_fmt(quat_of(Rc))}" fovy="{fv:.4f}"/>'


_maps = None
def distort(img_pinhole):
    """핀홀 렌더 → 실물 렌즈 왜곡(배럴) 적용. 왜곡 화면의 각 픽셀이 핀홀 화면 어디서 오는지 계산해 remap."""
    global _maps
    if _maps is None:
        uv = np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).astype(np.float32).reshape(-1, 1, 2)
        und = cv2.undistortPoints(uv, K, DIST, P=K).reshape(H, W, 2)     # 왜곡 픽셀 → 핀홀 픽셀
        _maps = (und[..., 0].astype(np.float32), und[..., 1].astype(np.float32))
    return cv2.remap(img_pinhole, _maps[0], _maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def side_by_side(real_bgr, sim_rgb, path, label=("실물", "시뮬")):
    sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
    both = np.hstack([real_bgr, sim_bgr]); blend = cv2.addWeighted(real_bgr, 0.5, sim_bgr, 0.5, 0)
    out = np.vstack([both, np.hstack([blend, np.zeros_like(blend)])])
    cv2.imwrite(path, out); return out


def jaw_features(img_rgb_or_bgr, sim):
    """턱 끝 2점과 아래 행의 바깥 가장자리 x 2개. sim=True면 노란 메시, False면 어두운 픽셀."""
    im = img_rgb_or_bgr.astype(int)
    if sim: mask = (im[..., 0] > 150) & (im[..., 1] > 150) & (im[..., 2] < 120)     # RGB 노랑
    else:
        g = im.mean(-1); mask = g < 70
    mask[:200] = False
    feats = []
    for half in (slice(0, W // 2), slice(W // 2, W)):
        sub = mask[:, half]; ys, xs = np.nonzero(sub)
        if len(ys) == 0: return None
        i = ys.argmin(); feats += [xs[i] + half.start, ys[i]]
        row = np.nonzero(sub[-1])[0]
        feats.append((row.min() if half.start == 0 else row.max()) + half.start if len(row) else (half.start if half.start == 0 else W - 1))
    return np.array(feats, float)


def fit_wrist(real_wr_bgr, grip=35.0, out=None):
    """격자 탐색: L(뒤로), H(턱 평면에서), tilt(턱 쪽 기울임), X(옆)로 실물 턱 끝 위치와 맞춤."""
    from so101_ik import Kin; k = Kin()
    target = jaw_features(real_wr_bgr, sim=False); print("실물 턱 특징 [좌끝x y 좌바깥x, 우끝x y 우바깥x]:", target)
    best = None
    grid = [(L, Hh, t, X) for L in (110, 130, 150, 170, 190, 210) for Hh in (20, 35, 50, 65, 80) for t in (0, 10, 20, 30, 40) for X in (-15, 0, 15)]
    for L, Hh, t, X in grid:
        s = PhysSim({}, kp=25, gripper_extra_xml=wrist_cam_xml(L, Hh, X, t))
        for nm in ("pad_fixed", "pad_moving"): s.m.geom(nm).rgba[3] = 0.0
        s.set_state(np.r_[HOME[:5], grip]); mujoco.mj_forward(s.m, s.d)
        f = jaw_features(s.render_cam("wrist"), sim=True)
        if f is None: continue
        c = float(np.sum((f - target) ** 2))
        if best is None or c < best[0]: best = (c, (L, Hh, t, X), f)
    print(f"최적: L {best[1][0]} H {best[1][1]} tilt {best[1][2]} X {best[1][3]}  특징 {best[2]}  오차 rms {np.sqrt(best[0]/6):.1f} px")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", nargs=2, default=["calib/cam_check_top.png", "calib/cam_check_wrist.png"])
    ap.add_argument("--wrist", nargs=3, type=float, default=[WRIST_DEFAULT["L"], WRIST_DEFAULT["H"], WRIST_DEFAULT["X"]]); ap.add_argument("--wrist-tilt", type=float, default=WRIST_DEFAULT["tilt"])
    ap.add_argument("--grip", type=float, default=35.0); ap.add_argument("--joints", nargs=6, type=float); ap.add_argument("--out", default="calib/sim/cam_compare"); ap.add_argument("--fit-wrist", action="store_true")
    a = ap.parse_args(); os.chdir(os.path.expanduser("~/robotics"))
    from cube_locate import locate
    real_top = cv2.imread(a.real[0]); real_wr = cv2.imread(a.real[1])
    if a.fit_wrist:
        b = fit_wrist(real_wr, a.grip); a.wrist = list(b[1][:2]) + [b[1][3]]; a.wrist_tilt = b[1][2]
    found, _ = locate(real_top)
    cubes = {n: (float(v["xy"][0]), float(v["xy"][1]), float(v.get("angle_deg", 0.0))) for n, v in found.items() if v}
    print("실물 사진 큐브:", {n: tuple(round(x, 1) for x in c) for n, c in cubes.items()})
    from so101_ik import Kin; k = Kin()
    wj = os.path.join(CAL, "wrist_cam.json")
    wx = __import__("json").load(open(wj))["mjcf"] if os.path.exists(wj) else wrist_cam_xml(a.wrist[0], a.wrist[1], a.wrist[2], a.wrist_tilt)
    print("손목 카메라:", "hand-eye 보정값" if os.path.exists(wj) else "임시 추정값")
    s = PhysSim(cubes, kp=25, extra_xml=top_cam_xml(k), gripper_extra_xml=wx)
    pose = np.array(a.joints) if a.joints else np.r_[HOME[:5], a.grip]
    s.set_state(pose); s.set_ctrl(pose); s.settle(0.5)
    top = distort(s.render_cam("top")); wr = s.render_cam("wrist")
    cv2.imwrite(a.out + "_top_sim.png", cv2.cvtColor(top, cv2.COLOR_RGB2BGR)); cv2.imwrite(a.out + "_wrist_sim.png", cv2.cvtColor(wr, cv2.COLOR_RGB2BGR))
    side_by_side(real_top, top, a.out + "_top.png"); side_by_side(real_wr, wr, a.out + "_wrist.png")
    # 큐브 픽셀 위치 비교 (탑): 실물 vs 시뮬 렌더에서 다시 locate
    sim_found, _ = locate(cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
    for n in cubes:
        r = found.get(n); sm = sim_found.get(n) if sim_found else None
        if r and sm: print(f"  {n}: 실물 픽셀 {tuple(round(x) for x in r['px'])} 격자 {tuple(round(float(x),1) for x in r['xy'])}  | 시뮬 픽셀 {tuple(round(x) for x in sm['px'])} 격자 {tuple(round(float(x),1) for x in sm['xy'])}")
        elif r: print(f"  {n}: 시뮬 렌더에서 검출 실패")
    print("저장:", a.out + "_top.png", a.out + "_wrist.png")
