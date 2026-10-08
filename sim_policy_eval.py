#!/usr/bin/env python3
"""실물로 학습한 정책(v4/v5)을 MuJoCo 안에서 실행해 real→sim 갭을 잰다. 정책은 시뮬 렌더(탑·손목, 왜곡 적용)와 관절값을 보고 30 Hz로 동작을 낸다.
사용:
  python ~/robotics/sim_policy_eval.py --model moonbeards/smolvla_color_cube_stack_unstack2_v4 --cond A --n 1 --save   # A 배치 1회, 화면 저장
  python ~/robotics/sim_policy_eval.py --model ... --cond A --n 10 ; --cond B --n 10                                   # 실물 평가와 같은 배치(classical_runs.csv의 A/B2)
  옵션: --steps 10 (재추론 간격, 실물 RTC execution_horizon 10과 맞춤) --dur 30 (초) --kp 25 (≈P16) --light
출력: 회차별 접근/잡기/들기/놓기 도달 여부, calib/sim/policy_<tag>_*.png, calib/sim_policy_runs.csv
"""
import os, sys, csv, time, argparse, json
import numpy as np, cv2, mujoco, torch
sys.path.insert(0, os.path.expanduser("~/robotics")); os.chdir(os.path.expanduser("~/robotics"))
from sim_phys import PhysSim, DEG_PER_PCT, CLOSED_READING
from sim_cam import top_cam_xml, distort
from sim_demo import wrist_xml, DEMO_REST, TASK, FPS
from so101_ik import Kin

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
RENAME = {"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}
DS_FEATURES = {"observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
               "observation.images.top": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
               "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
               "action": {"dtype": "float32", "shape": (6,), "names": JOINTS}}


def load_engine(model, device="mps", n_action_steps=10):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.rollout.inference.sync import SyncInferenceEngine
    cfg = PreTrainedConfig.from_pretrained(model); cfg.pretrained_path = model
    cfg.n_action_steps = n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(model, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=model, dataset_stats=None,
                                         preprocessor_overrides={"device_processor": {"device": device}, "rename_observations_processor": {"rename_map": RENAME}})
    keys = list(getattr(cfg, "action_feature_names", None) or JOINTS)
    eng = SyncInferenceEngine(policy, pre, post, DS_FEATURES, keys, TASK, device, "so_follower"); eng.start(); return eng, keys


def placements(cond):
    rows = list(csv.DictReader(open("calib/classical_runs.csv")))
    tag = {"A": "A", "B": "B2"}[cond]
    return [dict(orange=(float(r["ox"]), float(r["oy"]), float(r["cube_ang"] or 0)), red=(float(r["rx"]), float(r["ry"]), 0.0)) for r in rows if r["tag"] == tag]


IMG_AUG = None   # dict(noise=σ, blur=k, gain=±) 또는 None
def _augment(img, rng):
    a = IMG_AUG; x = img.astype(np.float32)
    if a.get("gain", 0): x = x * (1 + rng.uniform(-a["gain"], a["gain"]))
    if a.get("blur", 0) > 1: x = cv2.GaussianBlur(x, (a["blur"], a["blur"]), 0)
    if a.get("noise", 0): x = x + rng.normal(0, a["noise"], x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)

def shifted_wrist_xml(shift_mm):
    """손목 카메라를 시선 방향 반대로(턱에서 멀어지게) shift_mm 이동한 MJCF"""
    import re
    x = wrist_xml(); pos = np.array(re.search(r'pos="([^"]+)"', x).group(1).split(), float); q = np.array(re.search(r'quat="([^"]+)"', x).group(1).split(), float)
    R = np.empty(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
    pos = pos + R[:, 2] * shift_mm / 1000.0                       # MuJoCo 카메라는 −z를 봄 → +z가 뒤
    return re.sub(r'pos="[^"]+"', f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"', x)


def run_episode(eng, keys, cubes, kp, dur, light, save_prefix=None, rng=None, video=None, view=False, wrist_shift=0.0, state_mode="qpos", delay=0, hidden=None, cmd_delay=None):
    from lerobot.utils.feature_utils import build_dataset_frame
    k = Kin(); extra = top_cam_xml(k)
    if light:
        lp = (rng or np.random.default_rng()).uniform([0.0, -0.5, 0.9], [0.6, 0.2, 1.4]); di = 0.7
        extra += f'<light pos="{lp[0]:.2f} {lp[1]:.2f} {lp[2]:.2f}" dir="{-lp[0]+0.2:.2f} {-lp[1]:.2f} -1" diffuse="{di} {di} {di}"/>'
    comp = "auto" if hidden is None else ({"shoulder_lift": hidden, "elbow_flex": hidden * 0.5} if hidden > 0 else None)
    s = PhysSim(cubes, kp=kp, extra_xml=extra, gripper_extra_xml=shifted_wrist_xml(wrist_shift), compliance=comp)
    for nm in ("pad_fixed", "pad_moving"): s.m.geom(nm).rgba[3] = 0.0
    if cmd_delay is not None: s.delay = int(cmd_delay)
    s.set_state(DEMO_REST); s.set_ctrl(DEMO_REST); s.settle(0.3); eng.reset()
    o0, _ = s.cube_grid("orange"); r0, _ = s.cube_grid("red")
    stats = dict(min_tip_to_orange=1e9, first_contact_s=None, max_orange_z=o0[2], grip_closed_s=None, lifted_s=None, over_red_s=None, success=False, red_moved=0.0, min_grip_cmd=100.0, min_tip_z=1e9, trace=[])
    n = int(dur * FPS); t_inf = 0.0; vw = None
    if video:
        vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"avc1"), FPS, (1280, 960))
        cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135, -35, 0.7; cam.lookat[:] = k.grid_to_base((100, 170, 20)) / 1000.0
        ext = mujoco.Renderer(s.m, 480, 640)
    viewer = None
    if view:
        from mujoco import viewer as _mv; viewer = _mv.launch_passive(s.m, s.d); viewer.sync()
    from collections import deque
    cmd_hist = deque([DEMO_REST.copy()] * (delay + 1), maxlen=delay + 1)   # 실물 인코더는 백래시 앞에 있어 처짐이 안 보이고 명령을 133 ms 늦게 따라감
    for i in range(n):
        st = s.readings(); top = distort(s.render_cam("top")); wr = distort(s.render_cam("wrist"))
        if IMG_AUG: top, wr = _augment(top, rng), _augment(wr, rng)                    # 실물 카메라 질감 흉내(노이즈·흐림·노출 흔들림), --img-aug
        if state_mode == "encoder": st = np.r_[cmd_hist[0][:5], st[5]]          # 관절 5개는 지연된 명령, 그리퍼는 실제(큐브에 막힘이 보이게)
        vals = {jn: float(v) for jn, v in zip(JOINTS, st)}; vals["top"] = top; vals["wrist"] = wr
        frame = build_dataset_frame(DS_FEATURES, vals, "observation")
        t0 = time.time(); act = eng.get_action(frame); t_inf += time.time() - t0
        tgt = np.array([act[keys.index(jn)].item() for jn in JOINTS], float)
        s.tick(tgt); cmd_hist.append(tgt.copy())
        o, tilt = s.cube_grid("orange"); r_, _ = s.cube_grid("red"); tip = s.tip_grid(s.extra(35))
        d = np.linalg.norm(tip - o); stats["min_tip_to_orange"] = min(stats["min_tip_to_orange"], d)
        cont = s.contacts("orange"); tsec = (i + 1) / FPS
        if stats["first_contact_s"] is None and ("pad_fixed" in cont or "pad_moving" in cont): stats["first_contact_s"] = tsec
        if stats["grip_closed_s"] is None and tgt[5] < 20 and ("pad_fixed" in cont and "pad_moving" in cont): stats["grip_closed_s"] = tsec
        if o[2] > 35 and stats["lifted_s"] is None: stats["lifted_s"] = tsec
        if o[2] > 35 and np.linalg.norm(o[:2] - r_[:2]) < 25 and stats["over_red_s"] is None: stats["over_red_s"] = tsec
        stats["max_orange_z"] = max(stats["max_orange_z"], o[2]); stats["red_moved"] = float(np.linalg.norm(r_[:2] - r0[:2]))
        stats["min_grip_cmd"] = min(stats["min_grip_cmd"], float(tgt[5])); stats["min_tip_z"] = min(stats["min_tip_z"], float(tip[2]))
        if i % FPS == 0: stats["trace"].append((round(tsec), [round(float(v)) for v in tip], round(float(tgt[5])), round(float(st[5])), [round(float(v)) for v in o]))
        if save_prefix and i % 30 == 0: cv2.imwrite(f"{save_prefix}_{i//30:02d}s.png", cv2.cvtColor(np.hstack([top, wr]), cv2.COLOR_RGB2BGR))
        if vw is not None:
            ext.update_scene(s.d, cam); e = ext.render(); hud = np.zeros((480, 640, 3), np.uint8)
            for j, txt in enumerate([f"t {tsec:5.1f}s", f"grip cmd {tgt[5]:.0f} / actual {st[5]:.0f}", f"tip z {tip[2]:.0f} mm  orange z {o[2]:.0f}", f"contacts {cont}"]):
                cv2.putText(hud, txt, (10, 60 + 50 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            vw.write(cv2.cvtColor(np.vstack([np.hstack([top, wr]), np.hstack([e, hud])]), cv2.COLOR_RGB2BGR))
        if viewer is not None: viewer.sync(); time.sleep(1 / FPS)
        if abs(o[2] - 45) < 6 and np.linalg.norm(o[:2] - r_[:2]) < 10 and tilt < 10 and "red" in cont and not any(p in cont for p in ("pad_fixed", "pad_moving")):
            stats["success"] = True; stats["success_s"] = tsec; break
    o, tilt = s.cube_grid("orange"); r_, _ = s.cube_grid("red")
    if vw is not None: vw.release()
    if viewer is not None: viewer.close()
    stats.update(final_orange=[round(float(v), 1) for v in o], final_red=[round(float(v), 1) for v in r_], infer_s_per_tick=t_inf / n)
    if save_prefix: s.snapshot(f"{save_prefix}_final.png", lookat=(100, 170, 20))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="moonbeards/smolvla_color_cube_stack_unstack2_v4"); ap.add_argument("--cond", default="A"); ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10); ap.add_argument("--dur", type=float, default=30); ap.add_argument("--kp", type=float, default=None, help="미지정 시 sim_dynamics.json의 관절별 맞춤값"); ap.add_argument("--light", action="store_true"); ap.add_argument("--save", action="store_true"); ap.add_argument("--tag", default=None); ap.add_argument("--video", type=int, default=0, help="앞 N회차 mp4 저장"); ap.add_argument("--view", action="store_true", help="실시간 창 (mjpython 필요)"); ap.add_argument("--wrist-shift", type=float, default=0.0, help="손목 카메라를 뒤로 mm"); ap.add_argument("--state", default="qpos", help="qpos | encoder(지연된 명령을 상태로)"); ap.add_argument("--delay", type=int, default=4, help="encoder 모드 지연 프레임(실물 133 ms = 4)"); ap.add_argument("--cmd-delay", type=int, default=None, help="명령→모터 지연 프레임 강제(미지정=sim_dynamics.json)"); ap.add_argument("--kp-scale", type=float, default=1.0, help="맞춤 kp 배율(팔 5관절)"); ap.add_argument("--kp-grip-scale", type=float, default=1.0); ap.add_argument("--kp-joints", default=None, help="관절별 kp 직접 지정, 예 12,,,,12 (빈칸=맞춤값 유지)"); ap.add_argument("--kv-joints", default=None); ap.add_argument("--friction-scale", type=float, default=1.0); ap.add_argument("--stiff-cube", action="store_true", help="큐브 접촉을 단단하게(턱 판 파고듦 방지)"); ap.add_argument("--pcb", action="store_true"); ap.add_argument("--appearance", default=None); ap.add_argument("--wrist-cam", default=None); ap.add_argument("--img-aug", default=None, help="noise,blur,gain 예 6,3,0.15 (노이즈 σ, 흐림 커널, 노출 ±비율)"); ap.add_argument("--hidden", type=float, default=None, help="숨은 컴플라이언스 K_lift (N·m/rad, 팔꿈치는 절반). 0=없음, 미지정=sim_dynamics.json")
    a = ap.parse_args(); tag = a.tag or (a.model.split("_")[-1] + "_" + a.cond)
    if a.pcb:
        import sim_demo as _sd; _sd.WRIST_PCB = True
    if a.wrist_cam:
        import sim_demo as _sd2; _sd2.WRIST_CAM_PATH = a.wrist_cam
    if a.appearance:
        import sim_phys as _sp; _sp.apply_appearance(a.appearance)
    if a.img_aug:
        n_, b_, g_ = a.img_aug.split(","); globals()['IMG_AUG'] = dict(noise=float(n_), blur=int(b_), gain=float(g_))
    if a.stiff_cube: PhysSim.CUBE_CONTACT = PhysSim.STIFF_CUBE
    PhysSim.KP_SCALE = a.kp_scale; PhysSim.KP_GRIP_SCALE = a.kp_grip_scale; PhysSim.FRICTION_SCALE = a.friction_scale
    if a.kp_joints: PhysSim.KP_OVERRIDE = [float(x) if x.strip() else None for x in a.kp_joints.split(",")]
    if a.kv_joints: PhysSim.KV_OVERRIDE = [float(x) if x.strip() else None for x in a.kv_joints.split(",")]
    eng, keys = load_engine(a.model, n_action_steps=a.steps); print("정책 로드 완료, 동작 키", keys, flush=True)
    pl = placements(a.cond)[:a.n]; ok = 0; rng = np.random.default_rng(0)
    os.makedirs("calib/sim", exist_ok=True); f = "calib/sim_policy_runs.csv"; new = not os.path.exists(f)
    with open(f, "a") as fh:
        w = csv.writer(fh)
        if new: w.writerow(["time", "tag", "model", "cond", "i", "ox", "oy", "rx", "ry", "success", "first_contact_s", "grip_closed_s", "lifted_s", "over_red_s", "min_tip_mm", "max_orange_z", "red_moved_mm", "final_orange", "final_red"])
        for i, c in enumerate(pl):
            st = run_episode(eng, keys, c, a.kp, a.dur, a.light, save_prefix=(f"calib/sim/policy_{tag}_{i}" if a.save else None), rng=rng, video=(f"videos/sim_policy_{tag}_{i}.mp4" if i < a.video else None), view=a.view, wrist_shift=a.wrist_shift, state_mode=a.state, delay=a.delay, hidden=a.hidden, cmd_delay=a.cmd_delay)
            ok += int(st["success"])
            print(f"[{i+1}/{len(pl)}] 주황 {c['orange'][:2]} 빨강 {c['red'][:2]} → {'성공' if st['success'] else '실패'} | 첫 접촉 {st['first_contact_s']}s 잡음 {st['grip_closed_s']}s 들림 {st['lifted_s']}s 빨강 위 {st['over_red_s']}s | 최소거리 {st['min_tip_to_orange']:.0f} mm 최대 z {st['max_orange_z']:.0f} 빨강 이동 {st['red_moved']:.0f} mm | 그리퍼 명령 최소 {st['min_grip_cmd']:.0f} 손끝 최저 z {st['min_tip_z']:.0f} | 추론 {st['infer_s_per_tick']*1000:.0f} ms/틱", flush=True)
            print("    1초마다 (t, 손끝 xyz, 그리퍼 명령, 실제, 주황 xyz):", st["trace"][:16], flush=True)
            w.writerow([time.strftime("%m%d_%H%M%S"), tag, a.model, a.cond, i, *[round(v, 1) for v in c["orange"][:2]], *[round(v, 1) for v in c["red"][:2]], int(st["success"]), st["first_contact_s"], st["grip_closed_s"], st["lifted_s"], st["over_red_s"], round(st["min_tip_to_orange"], 1), round(st["max_orange_z"], 1), round(st["red_moved"], 1), st["final_orange"], st["final_red"]]); fh.flush()
    print(f"결과 {tag}: {ok}/{len(pl)}")
