#!/usr/bin/env python3
"""MuJoCo 3단계: 시연 자동 생성. 무작위 큐브 배치 → 고전 계획(IK)으로 쌓기 → 탑·손목 화면 + 관절값을 30 Hz로 LeRobot 데이터셋에 저장.
사용:
  python ~/robotics/sim_demo.py --n 2 --preview          # 2회 생성, 첫 회차 프레임을 calib/sim/demo_preview_*.png 로 저장 (데이터셋은 sim_stack_test)
  python ~/robotics/sim_demo.py --n 200 --repo moonbeards/sim_stack_v1 --seed 1
  옵션: --kp 25  --keep-failures  --range mix|real|wide|demo (mix=절반은 실물 시연 배치 분포+지터, 절반은 도달 범위 전체)  --light (조명 무작위)
저장 형식: 실물 데이터셋과 동일(observation.state/action 6, observation.images.top/wrist 480×640 h264, task 문장). 실패 회차는 기본 제외.
"""
import os, sys, re, argparse, time, json, shutil
WRIST_CAM_PATH = os.path.join(os.path.expanduser('~/robotics/calib'), 'wrist_cam.json')   # --wrist-cam 으로 교체(v3: calib/wrist_cam_v3.json 큐브 맞춤 카메라)
WRIST_PCB = False   # True면 손목 카메라 기판 geom 추가(v3 생성 조건). sim_demo --pcb / sim_policy_eval --pcb
import numpy as np, cv2, mujoco
sys.path.insert(0, os.path.expanduser("~/robotics"))
from sim_phys import PhysSim, HOME, DEG_PER_PCT, CLOSED_READING
from sim_cam import top_cam_xml, wrist_cam_xml, distort, CAL
from so101_ik import Kin

TASK = "Pick up the orange cube and place it on top of the red cube"
FPS = 30
DEMO_REST = np.array([-2.0, 6.0, 97.0, 66.0, -89.0, 26.0])     # 실물 시연 176회의 시작 자세(에피소드 0 첫 프레임)
HOVER, GRASP_Z, OPEN, CLOSE = 60.0, 3.0, 28.0, 0.0      # 실물 시연: 접근 중 그리퍼 명령 중앙값 28, 문 상태 0
PLACE_Z = 50.0                                          # 놓기 직전 주황 중심 높이(빨강 위 45 + 5 mm 여유; 2 mm 이하로 띄우면 손안에서 기운 큐브 모서리가 빨강을 밀어 y −4 mm 치우침)
MAX_DEG_S = 25.0
RANGES = {"demo": dict(ox=(60, 130), oy=(140, 200), rx=(80, 160), ry=(150, 220)),
          "wide": dict(ox=(-60, 220), oy=(120, 260), rx=(-60, 220), ry=(120, 260))}
# 실물 시연 통계(calib/real_demo_motion_stats.json, real_demo_placements.json): 있으면 시작 자세·정지 시간·속도·배치를 그 분포에서 뽑음
_MS = json.load(open(os.path.join(CAL, "real_demo_motion_stats.json"))) if os.path.exists(os.path.join(CAL, "real_demo_motion_stats.json")) else None
_PL = json.load(open(os.path.join(CAL, "real_demo_placements.json"))) if os.path.exists(os.path.join(CAL, "real_demo_placements.json")) else None
REST_MED = np.array([-2.5, -4.2, 89.3, 75.3, -90.3, 26.9]); REST_STD = np.array([2.5, 7.0, 5.0, 6.4, 4.5, 5.3])   # 실물 쌓기 시연 124회 시작 자세


def wrist_xml(pcb=None):
    pcb = WRIST_PCB if pcb is None else pcb
    """손목 카메라 MJCF(hand-eye 결과). pcb=True면 실물처럼 카메라 뒤에 붙은 기판(약 30×32 mm, 어두운 회색)을 시각용 geom으로 추가 —
    탑 카메라에서 그리퍼 끝을 덮는 큰 사각이 보이고 손안 큐브를 가린다(10-01 실물 비교)."""
    wj = os.path.expanduser(WRIST_CAM_PATH)
    x = json.load(open(wj))["mjcf"] if os.path.exists(wj) else wrist_cam_xml()
    if not pcb or "pos=" not in x: return x
    pos = np.array(re.search(r'pos="([^"]+)"', x).group(1).split(), float); q = np.array(re.search(r'quat="([^"]+)"', x).group(1).split(), float)
    R = np.empty(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3); p = pos + R[:, 2] * 0.004       # 렌즈 4 mm 뒤
    return x + f'\n    <geom name="wrist_pcb" type="box" size="0.015 0.016 0.0008" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" rgba="0.07 0.075 0.08 1" contype="0" conaffinity="0"/>'


class DemoGen:
    def __init__(self, kp=None, light=False, rng=None, rand=False, dart=0.0):
        self.k = Kin(); self.kp = kp; self.light = light; self.rand = rand; self.rng = rng or np.random.default_rng(0)
        self.dart = dart   # DART식 교란: 접근·운반 구간에서 실행 명령에 관절 잡음(표준편차 °)을 주되 기록되는 action은 계획기의 깨끗한 명령 → '빗나간 상태에서 바로잡는' 데이터

    def new_scene(self, cubes):
        import sim_phys, re
        if self.rand:   # 도메인 무작위화: 실물 사진 통계 주변에서 흔듦
            base = json.load(open(os.path.expanduser(getattr(sim_phys, "APPEARANCE_PATH", os.path.join(CAL, "sim_appearance.json")))))   # --appearance로 바꾼 파일을 따름
            tb = float(np.clip(base["tb"] + self.rng.normal(0, 0.04), 0.5, 0.85)); sim_phys.TABLE_RGBA = f"{tb:.3f} {tb:.3f} {tb:.3f} 1"
            def jit(c, s_=0.05):
                v = np.array(c.split()[:3], float) * (1 + self.rng.normal(0, s_, 3)); return " ".join(f"{x:.3f}" for x in np.clip(v, 0, 1)) + " 1"
            sim_phys.CUBE_RGBA = {"orange": jit(base["oc"]), "red": jit(base["rc"])}
            sim_phys.LIGHT.update(head_diffuse=float(np.clip(base["hd"] * self.rng.uniform(0.75, 1.3), 0.1, 0.8)), ambient=float(np.clip(base["amb"] * self.rng.uniform(0.75, 1.3), 0.1, 0.8)),
                                  diffuse=float(np.clip(base["dif"] * self.rng.uniform(0.5, 1.6), 0.0, 0.8)), diffuse2=float(np.clip(base["dif2"] * self.rng.uniform(0.5, 1.6), 0.0, 0.8)),
                                  pos=" ".join(f"{x:.2f}" for x in self.rng.uniform([-0.1, -0.6, 0.9], [0.7, 0.3, 1.5])))
            sim_phys.CUBE = float(np.clip(0.030 + self.rng.normal(0, 0.0007), 0.028, 0.032))
        extra = top_cam_xml(self.k)
        if self.rand:   # 탑 카메라 자세 지터 ±2 mm, ±0.5°
            pos = np.array(re.search(r'pos="([^"]+)"', extra).group(1).split(), float) + self.rng.normal(0, 0.002, 3)
            q = np.array(re.search(r'quat="([^"]+)"', extra).group(1).split(), float); dq = np.r_[1.0, np.deg2rad(self.rng.normal(0, 0.5, 3)) / 2]; dq /= np.linalg.norm(dq)
            q2 = np.empty(4); mujoco.mju_mulQuat(q2, q, dq)
            extra = re.sub(r'pos="[^"]+"', f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"', extra, count=1); extra = re.sub(r'quat="[^"]+"', f'quat="{q2[0]:.6f} {q2[1]:.6f} {q2[2]:.6f} {q2[3]:.6f}"', extra, count=1)
        if self.light:
            lp = self.rng.uniform([0.0, -0.5, 0.9], [0.6, 0.2, 1.4]); di = self.rng.uniform(0.5, 0.9)
            extra += f'<light pos="{lp[0]:.2f} {lp[1]:.2f} {lp[2]:.2f}" dir="{-lp[0]+0.2:.2f} {-lp[1]:.2f} -1" diffuse="{di:.2f} {di:.2f} {di:.2f}"/>'
        wx = wrist_xml()
        if self.rand:   # 손목 카메라 지터 ±1 mm, ±0.5°
            pos = np.array(re.search(r'pos="([^"]+)"', wx).group(1).split(), float) + self.rng.normal(0, 0.001, 3)
            q = np.array(re.search(r'quat="([^"]+)"', wx).group(1).split(), float); dq = np.r_[1.0, np.deg2rad(self.rng.normal(0, 0.5, 3)) / 2]; dq /= np.linalg.norm(dq); q2 = np.empty(4); mujoco.mju_mulQuat(q2, q, dq)
            wx = re.sub(r'pos="[^"]+"', f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"', wx, count=1); wx = re.sub(r'quat="[^"]+"', f'quat="{q2[0]:.6f} {q2[1]:.6f} {q2[2]:.6f} {q2[3]:.6f}"', wx, count=1)
        s = PhysSim(cubes, kp=self.kp, extra_xml=extra, gripper_extra_xml=wx)
        for nm in ("pad_fixed", "pad_moving"): s.m.geom(nm).rgba[3] = 0.0     # 접촉면은 보이지 않게(실물엔 없음)
        return s

    def roll_for_cube(self, s, p, grip, ang_deg):
        best = None
        for th in (ang_deg, ang_deg + 90.0):
            d = (np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)), 0.0)
            b = s.roll_for_open_dir(p, grip, d)
            if b and (best is None or b[0] > best[0]): best = b
        return best

    def plan(self, s):
        """(이름, 목표 읽기값6, 소요 초) 목록. 큐브 위치는 시뮬 상태에서 직접 읽음(완전 정보)."""
        o, oa = s.cube_grid("orange"); r_, ra = s.cube_grid("red")
        oang = float(np.rad2deg(np.arctan2(*(s.k.R.T @ s.d.xmat[s.m.body("orange").id].reshape(3, 3)[:, 0])[[1, 0]])))
        rang = float(np.rad2deg(np.arctan2(*(s.k.R.T @ s.d.xmat[s.m.body("red").id].reshape(3, 3)[:, 0])[[1, 0]])))
        b = self.roll_for_cube(s, (o[0], o[1], GRASP_Z + HOVER), OPEN, oang)
        if b is None: return None
        roll = b[1]; r30 = s.reading_for_pad_gap(30.0)
        steps = []
        def ik(p, g, q0=None):
            q, e = s.ik(p, roll, g, q0=q0); return q, e
        # 열림 방향(격자) 부호 포함: 움직이는 턱이 +n 쪽. 실물 정책은 큐브가 움직이는 턱 쪽으로 ~15 mm 치우친 채 z 10까지 내려온 뒤 맞추며 잡음
        gg = s.gg; lever = gg.lever / np.linalg.norm(gg.lever); n_open = np.cross(gg.axis / np.linalg.norm(gg.axis), lever)
        q_tmp, _ = ik((o[0], o[1], GRASP_Z + HOVER), OPEN)
        q_true = s.k.urdf_to_reading(s.k.reading_to_urdf(q_tmp) + np.array([0, 0, 0, 0, s.ROLL_FIX_DEG])); _, Rm = s.k.fk_base(q_true); n_w = s.k.R.T @ (Rm @ (gg.Rf @ n_open)); n_w[2] = 0; n_w /= np.linalg.norm(n_w)
        off = float(self.rng.uniform(8, 20)) if _MS else 0.0; p_side = np.array([o[0], o[1], 0.0]) - off * n_w
        q1, e1 = ik((p_side[0], p_side[1], GRASP_Z + 40.0), OPEN); q1b, e1b = ik((p_side[0], p_side[1], 10.0), OPEN, q1); q2, e2 = ik((o[0], o[1], GRASP_Z), OPEN, q1b)
        if max(e1, e1b, e2) > 3: return None
        steps += [("주황 옆 대기", np.r_[q1, OPEN]), ("주황 옆 낮게", np.r_[q1b, OPEN]), ("주황 맞추기", np.r_[q2, OPEN]), ("닫기", np.r_[q2, CLOSE]), ("들어올리기", np.r_[q1, CLOSE])]
        self._place = (r_, rang, roll, r30); return steps

    def plan_place(self, s, in_hand):
        r_, rang, roll, r30 = self._place
        p_hover = (r_[0] - in_hand[0], r_[1] - in_hand[1], 45.0 + 2.0 - in_hand[2] + 40.0)
        q3, e3 = s.ik(p_hover, roll, r30)
        if e3 > 3: return None
        return [("빨강 위 대기", np.r_[q3, CLOSE])]

    def plan_place2(self, s, q3, corr):
        """빨강 위 대기 후 실제 손안 큐브 위치를 다시 재서(처짐 반영) 놓기 목표를 보정"""
        r_, rang, roll, r30 = self._place
        p_place = (r_[0] - corr[0], r_[1] - corr[1], PLACE_Z - corr[2])      # 기준점 = 빨강 − 손안 오프셋 → 큐브가 빨강 위 PLACE_Z(놓기 높이에서 처짐이 더 커지므로 여유 4 mm)
        q4, e4 = s.ik(p_place, roll, r30, q0=q3)
        if e4 > 3: return None
        return [("놓기 높이", np.r_[q4, CLOSE]), ("열기", np.r_[q4, OPEN]), ("올리기", np.r_[q3, OPEN]), ("복귀", self._rest.copy())]

    def run_episode(self, cubes, record=None, preview=None, verbose=False):
        """record: callable(frame dict) 또는 None. 반환 (성공 여부, 프레임 수, 정보)."""
        rest = np.clip(REST_MED + self.rng.normal(0, 1, 6) * REST_STD, [-30, -30, 60, 40, -120, 15], [30, 20, 110, 100, -60, 40]) if _MS else DEMO_REST.copy()
        self._rest = rest
        s = self.new_scene(cubes); s.set_state(rest); s.set_ctrl(rest); s.settle(0.3)
        steps = self.plan(s)
        if steps is None: return False, 0, "IK 불가"
        frames = [0]; cur = rest.copy(); in_hand = None; protect = [False]
        def tick(target, exec_=None):
            nonlocal cur
            s.tick(target if exec_ is None else exec_)                        # 실행은 교란된 명령, 기록은 깨끗한 명령
            st = s.readings().astype(np.float32); act = np.asarray(target, np.float32)
            if record is not None or preview is not None:
                top = distort(s.render_cam("top")); wr = distort(s.render_cam("wrist"))
                if record is not None: record({"observation.state": st, "action": act, "observation.images.top": top, "observation.images.wrist": wr, "task": TASK})
                if preview is not None and frames[0] % 15 == 0:
                    cv2.imwrite(f"{preview}_{frames[0]:04d}.png", cv2.cvtColor(np.hstack([top, wr]), cv2.COLOR_RGB2BGR))
            frames[0] += 1
        def move(target, hold=0.4, min_dur=0.6, phase="app"):
            nonlocal cur
            r0 = cur.copy(); rt = np.asarray(target, float)
            if _MS:   # 실물 단계별 최고 속도(90분위) 분포에서 뽑음; 코사인 보간의 최고 속도 = (π/2)·Δq/시간
                v = float(self.rng.choice([x for x in _MS[{"app": "v_app", "tr": "v_tr", "ret": "v_ret"}[phase]] if x > 20]))
                dur = max(min_dur, (np.pi / 2) * np.abs(rt[:5] - r0[:5]).max() / v)
            else:
                dur = max(min_dur, np.abs(rt[:5] - r0[:5]).max() / MAX_DEG_S)
            n = int(round(dur * FPS))
            d = np.zeros(6)
            if self.dart > 0 and phase in ("app", "tr") and self.rng.random() < 0.6 and not protect[0]:
                d[:4] = self.rng.normal(0, self.dart, 4)                              # pan·lift·elbow·wflex 교란(롤·그리퍼 제외)
            for i in range(1, n + 1):
                c = r0 + (rt - r0) * (0.5 - 0.5 * np.cos(np.pi * i / n)); tick(c, c + d * min(1.0, 2.0 * i / n) if d.any() else None)   # 교란은 구간 전반부에 서서히 들어감
            for _ in range(int(hold * FPS)): tick(rt, rt + d if d.any() else None)
            cur = rt + d                                                              # 다음 구간은 실제(교란된) 자세에서 깨끗한 목표로 → 복구 동작이 기록됨
            if verbose and d.any(): print(f"    교란 {np.round(d[:4],1)}°")
        start_pause = float(np.clip(self.rng.choice(_MS["start_pause"]), 0.2, 3.6)) if _MS else 0.5
        for _ in range(int(start_pause * FPS)): tick(rest)          # 실물처럼 시작 정지(중앙값 2.9 s)
        for name, tgt in steps:
            protect[0] = name in ("주황 맞추기", "닫기", "들어올리기")                       # 정밀 구간·들어올리기는 교란 없음
            move(tgt, hold=(0.5 if name in ("닫기",) else 0.2), phase=("tr" if name == "들어올리기" else "app"))
            if verbose: print(f"    {name}: 주황 {np.round(s.cube_grid('orange')[0],1)} 빨강 {np.round(s.cube_grid('red')[0],1)} 접촉(주황) {s.contacts('orange')} 접촉(빨강) {s.contacts('red')}")
        o, _ = s.cube_grid("orange"); tip = s.tip_grid(s.extra(self._place[3])); in_hand = o - tip
        if o[2] < 35: return False, frames[0], "들어올리기 실패"
        steps2 = self.plan_place(s, in_hand)
        if steps2 is None: return False, frames[0], "놓기 IK 불가"
        move(steps2[0][1], hold=0.3, phase="tr")
        if verbose: print(f"    빨강 위 대기: 주황 {np.round(s.cube_grid('orange')[0],1)}")
        tip = s.tip_grid(s.extra(self._place[3])); o_now, _ = s.cube_grid("orange"); corr = (o_now[0] - tip[0], o_now[1] - tip[1], o_now[2] - tip[2])   # 손안 큐브 오프셋(대기 자세, 처짐 포함)
        rest_steps = self.plan_place2(s, steps2[0][1][:5], corr)
        if rest_steps is None: return False, frames[0], "놓기 IK 불가"
        r_, rang, roll, r30 = self._place
        for name, tgt in rest_steps:
            protect[0] = name in ("놓기 높이", "열기")
            move(tgt, hold=(0.4 if name == "열기" else 0.2), phase=("ret" if name in ("올리기", "복귀") else "tr"))
            if name == "놓기 높이":      # 실물 파이프라인의 정착 보정처럼: 도착 후 큐브가 빨강 중심에서 벗어난 만큼 한 번 더 이동
                p_cmd = None
                for _ in range(4):
                    o_now, _ = s.cube_grid("orange"); err = o_now[:2] - np.array(r_[:2]); dz = PLACE_Z - o_now[2]   # xy 오차와 높이 오차(빨강을 눌러 밀지 않도록 2~4 mm 띄움)
                    if np.linalg.norm(err) < 1.0 and abs(dz) < 1.5: break
                    if p_cmd is None: p_cmd = np.array(s.tip_grid(s.extra(r30)), float)
                    p_cmd = p_cmd + np.array([-err[0], -err[1], dz])      # 명령 누적(적분): 마찰 사구간 때문에 2~3 mm 명령엔 팔이 안 움직이므로 오차가 남는 만큼 계속 더한다
                    p_fix = tuple(p_cmd); qf, ef = s.ik(p_fix, roll, r30, q0=tgt[:5])
                    if ef > 3: break
                    move(np.r_[qf, CLOSE], hold=0.2, min_dur=0.4, phase="tr")
                if verbose: print(f"    놓기 보정 후: 주황 {np.round(s.cube_grid('orange')[0],1)} 빨강 {np.round(r_,1)}")
            if verbose: print(f"    {name}: 주황 {np.round(s.cube_grid('orange')[0],1)} 빨강 {np.round(s.cube_grid('red')[0],1)} 접촉(주황) {s.contacts('orange')} 접촉(빨강) {s.contacts('red')}")
        end_pause = float(np.clip(self.rng.choice(_MS["end_pause"]), 0.2, 3.6)) if _MS else 2.0
        for _ in range(int(end_pause * FPS)): tick(rest)          # 실물 시연처럼 끝 정지(중앙값 0.7 s)
        o, tilt = s.cube_grid("orange"); r_, _ = s.cube_grid("red")
        ok = abs(o[2] - 45) < 6 and np.linalg.norm(o[:2] - r_[:2]) < 10 and tilt < 10
        return ok, frames[0], f"주황 {np.round(o,1)} 빨강 {np.round(r_,1)} 기울기 {tilt:.0f}°"

    def sample_cubes(self, mode="demo"):
        if mode in ("real", "mix") and _PL:
            if mode == "real" or self.rng.random() < 0.5:
                for _ in range(50):
                    o = np.array(_PL["orange_grid_mm"][self.rng.integers(len(_PL["orange_grid_mm"]))][:2]) + self.rng.normal(0, 6, 2)
                    r = np.array(_PL["red_grid_mm"][self.rng.integers(len(_PL["red_grid_mm"]))][:2]) + self.rng.normal(0, 6, 2)
                    if np.hypot(*(o - r)) >= 60: return {"orange": (float(o[0]), float(o[1]), float(self.rng.uniform(-30, 30))), "red": (float(r[0]), float(r[1]), float(self.rng.uniform(-30, 30)))}
            mode = "wide"
        R = RANGES[mode]
        for _ in range(100):
            o = (self.rng.uniform(*R["ox"]), self.rng.uniform(*R["oy"])); r = (self.rng.uniform(*R["rx"]), self.rng.uniform(*R["ry"]))
            if np.hypot(o[0] - r[0], o[1] - r[1]) >= 60 and np.hypot(o[0] - 87.5, o[1] + 92.3) < 330 and np.hypot(r[0] - 87.5, r[1] + 92.3) < 330:
                return {"orange": (o[0], o[1], float(self.rng.uniform(-30, 30))), "red": (r[0], r[1], float(self.rng.uniform(-30, 30)))}
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2); ap.add_argument("--repo", default="moonbeards/sim_stack_test"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kp", type=float, default=None); ap.add_argument("--keep-failures", action="store_true"); ap.add_argument("--range", default="mix"); ap.add_argument("--light", action="store_true"); ap.add_argument("--rand", action="store_true", help="도메인 무작위화(조명·책상·큐브 색/크기·카메라 자세)"); ap.add_argument("--soft-cube", action="store_true", help="예전 무른 큐브 접촉(sim_stack_v1 생성 조건)"); ap.add_argument("--dart", type=float, default=0.0, help="DART식 교란 표준편차(°), 예 2"); ap.add_argument("--pcb", action="store_true", help="손목 카메라 기판 geom(v3)"); ap.add_argument("--wrist-cam", default=None, help="손목 카메라 json(예 calib/wrist_cam_v3.json)"); ap.add_argument("--appearance", default=None, help="외관 json(예 calib/sim_appearance_v3.json)"); ap.add_argument("--kp-joints", default=None, help="관절별 kp 덮어쓰기, 예 16,,,,16"); ap.add_argument("--kv-joints", default=None); ap.add_argument("--preview", action="store_true"); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(); os.chdir(os.path.expanduser("~/robotics"))
    from sim_phys import PhysSim
    import sim_phys as _sp
    if a.pcb: globals()['WRIST_PCB'] = True
    if a.wrist_cam: globals()['WRIST_CAM_PATH'] = a.wrist_cam
    if a.appearance: _sp.apply_appearance(a.appearance)
    if not a.soft_cube: PhysSim.CUBE_CONTACT = PhysSim.STIFF_CUBE
    if a.kp_joints: PhysSim.KP_OVERRIDE = [float(x) if x.strip() else None for x in a.kp_joints.split(",")]
    if a.kv_joints: PhysSim.KV_OVERRIDE = [float(x) if x.strip() else None for x in a.kv_joints.split(",")]     # 10-01: 단단한 큐브(쥐었을 때 읽기값 13.5 ≈ 실물 13.6, 미끄러짐 없음)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{a.repo}")
    if os.path.exists(root): shutil.rmtree(root)
    feats = {"action": {"dtype": "float32", "shape": (6,), "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]},
             "observation.state": {"dtype": "float32", "shape": (6,), "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]},
             "observation.images.top": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
             "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}}
    ds = LeRobotDataset.create(a.repo, FPS, feats, root=root, robot_type="so_follower", rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=30))
    gen = DemoGen(kp=a.kp, light=a.light, rng=np.random.default_rng(a.seed), rand=a.rand, dart=a.dart); ok_n = 0; t0 = time.time(); log = []
    for i in range(a.n):
        cubes = gen.sample_cubes(a.range); t1 = time.time()
        ok, nf, info = gen.run_episode(cubes, record=ds.add_frame, preview=(f"calib/sim/demo_preview" if (a.preview and i == 0) else None), verbose=a.verbose)
        if ok or a.keep_failures: ds.save_episode(); ok_n += int(ok)
        else: ds.clear_episode_buffer()
        log.append(dict(i=i, cubes=cubes, ok=bool(ok), frames=int(nf), info=str(info)))
        print(f"[{i+1}/{a.n}] 주황 {tuple(round(v,1) for v in cubes['orange'])} 빨강 {tuple(round(v,1) for v in cubes['red'])} → {'성공' if ok else '실패'} {nf/FPS:.1f}s {info}  ({time.time()-t1:.0f}s)", flush=True)
    ds.finalize(); json.dump({"args": vars(a), "cube_contact": PhysSim.CUBE_CONTACT, "episodes": log}, open(os.path.join(root, "gen_log.json"), "w"), ensure_ascii=False, indent=1)
    print(f"완료: 성공 {ok_n}/{a.n}, 저장 {ds.meta.total_episodes}회 {ds.meta.total_frames}프레임, {(time.time()-t0)/60:.1f}분 → {root}")
