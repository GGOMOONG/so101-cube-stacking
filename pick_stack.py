#!/usr/bin/env python3
"""고전 파이프라인 쌓기: 탑 카메라로 주황·빨강 위치를 찾고(cube_locate), 역기구학(so101_ik)으로 주황을 집어 빨강 위에 놓는다.
사용: python ~/robotics/pick_stack.py            (카메라 0)
      python ~/robotics/pick_stack.py --dry      (측정과 계획만, 팔은 안 움직임)
      python ~/robotics/pick_stack.py --dry --fake 90 160 130 200   (카메라 없이 좌표를 직접 넣어 계획만)
      python ~/robotics/pick_stack.py --auto --tag A                (Enter 없이 자동 진행, 끝에 결과 입력 → calib/classical_runs.csv)
손목 회전: 후보 −50/−5/40/−95° 계획을 MuJoCo(sim_check)로 훑어 빨강 큐브와 여유가 가장 큰 것을 자동 선택.
절차: 카메라 창에서 두 큐브가 잡히면 스페이스 → 계획 출력 → Enter로 승인 → 실행. 각 단계 전 Enter, Ctrl+C로 정지.
단계: 주황 위 대기(+60) → 열고 내려가 잡기 높이(큐브 중심 15 mm) → 닫기 → 올리기 → 빨강 위 대기 → 내려가 놓기 높이(빨강 윗면 30 + 15 + 여유 5) → 열기 → 올리기.
"""
import sys, os, time, json
import numpy as np
import cv2
sys.path.insert(0, os.path.expanduser("~/robotics"))
from so101_ik import Kin
import cube_locate as CL

PORT = "/dev/tty.usbmodem5B790156431"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOVER = 60.0
GRASP_Z = 18.0          # 큐브 중심(15)보다 3 mm 위: 닫을 때 팔이 5~8 mm 내려앉아도 패드가 테이블에 닿지 않게 (09-29 B-2)
PLACE_Z = 30.0 + 15.0 + 3.0   # 빨강 윗면 + 큐브 반높이 + 여유 3 mm (09-29: 5 mm는 높다고 봄)
ROLL_URDF = -50.0
GRIP_OPEN, GRIP_CLOSE = 35.0, 12.0  # 09-29 실측: 패드 간격 = 끝 간격 − 20. 35 → 패드 약 52 mm, 30 mm 큐브는 읽기값 ≈ 21에서 닿음 → 12로 조여 물기
MAX_DEG_PER_S = 25.0
Y_BIAS = 2.0            # 09-28 이동 시험: 세 점 모두 y −2 mm → 보정
DRY = "--dry" in sys.argv
AUTO = "--auto" in sys.argv                      # 단계 사이 Enter 없이 자동 진행(막힘 정지는 유지). 10회 측정용
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""   # 실험 조건 표시(A/B/C)
RUN_LOG = os.path.expanduser("~/robotics/calib/classical_runs.csv")


def ask(prompt):
    if AUTO: print(prompt + " (자동)"); return ""
    return input(prompt)


def measure(cam_idx=0):
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
    print("두 큐브가 색 테두리로 잡히면 스페이스, q 취소", flush=True)
    out = None
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        o, vis = CL.locate(f)
        cv2.imshow("pick_stack: measure", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and o.get("orange") and o.get("red"):
            out = o; cv2.imwrite(os.path.expanduser(time.strftime("~/robotics/calib/pick_%m%d_%H%M%S.png")), f); break
        if k == ord("q"):
            break
    cv2.destroyAllWindows()
    global CAP; CAP = cap          # 단계별 사진용으로 열어 둠
    return out


CAP = None; WRIST = None
WRIST_IDX = 1                       # 손목 카메라 인덱스(카메라 검사로 확인)
RUN_TAG = time.strftime("%m%d_%H%M%S")
def snap(label):
    """탑(0)·손목(1) 카메라 현재 프레임 저장(버퍼 비우려 5장 읽음) → calib/pick_<시각>_<단계>_top/wrist.png"""
    global WRIST
    if WRIST is None and not DRY:
        WRIST = cv2.VideoCapture(WRIST_IDX, cv2.CAP_AVFOUNDATION)
        WRIST.set(cv2.CAP_PROP_FRAME_WIDTH, 640); WRIST.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); WRIST.set(cv2.CAP_PROP_FPS, 30)
    for cap, tag in [(CAP, "top"), (WRIST, "wrist")]:
        if cap is None: continue
        f = None
        for _ in range(5):
            ok, fr = cap.read()
            if ok: f = fr
        if f is not None:
            p = os.path.expanduser(f"~/robotics/calib/pick_{RUN_TAG}_{label}_{tag}.png"); cv2.imwrite(p, f); print(f"    사진 {p}")


if "--fake" in sys.argv:   # 카메라 없이 계획만 시험: --fake ox oy rx ry
    i = sys.argv.index("--fake"); v = [float(x) for x in sys.argv[i + 1:i + 5]]
    o = {"orange": {"xy": (v[0], v[1])}, "red": {"xy": (v[2], v[3])}}
else:
    o = measure()
if o is None:
    sys.exit("측정 취소 또는 큐브 미검출")
ox, oy = o["orange"]["xy"]; rx, ry = o["red"]["xy"]
oy += Y_BIAS; ry += Y_BIAS
print(f"주황 ({ox:.1f}, {oy:.1f})  빨강 ({rx:.1f}, {ry:.1f})  [y 보정 +{Y_BIAS}]")

k = Kin()
from gripper_geom import GripperGeom
GG = GripperGeom()
# 잡기 중심 실측 보정(grasp_center_calib.py): 계산 기준점 → 실제 잡기 중심 오프셋(그리퍼 x,y), 계산 턱 축 → 실제 턱 축 각도
GC_PATH = os.path.expanduser("~/robotics/calib/grasp_center.json")
GC = json.load(open(GC_PATH)) if os.path.exists(GC_PATH) else {"pad_offset_mm": [0.0, 0.0], "jaw_angle_offset_deg": 0.0}
PAD = np.array([GC["pad_offset_mm"][0], GC["pad_offset_mm"][1], 0.0])
JAW_ANG_OFF = float(GC.get("jaw_angle_offset_deg") or 0.0)
print(f"잡기 중심 보정: 패드 오프셋 {np.round(PAD[:2],1)} mm, 턱 각도 오프셋 {JAW_ANG_OFF:.1f}°  ({'실측' if os.path.exists(GC_PATH) else '미실측, 0 가정'})")
EXTRA_OPEN = GG.mid_offset_mm(GRIP_OPEN) + PAD                  # 열린 두 턱의 중간점 + 패드 중앙 → 접근·잡기 목표
EXTRA_HOLD = GG.mid_offset_mm(GG.reading_for_pad_gap(30.0)) + PAD   # 30 mm 큐브를 문 상태의 큐브 중심 → 들기·놓기 목표
print(f"그리퍼 기하: 열림 {GRIP_OPEN:.0f}% 간격 {GG.gap_mm(GRIP_OPEN):.0f} mm, 중간점 오프셋 {np.round(EXTRA_OPEN,1)} / 물었을 때 {np.round(EXTRA_HOLD,1)} mm")
def jaw_angle_deg(q5):
    """잡기 자세에서 실제 턱이 닫히는 축이 격자 x축과 이루는 각(도, 수평면 투영) = 계산 축 + 실측 오프셋"""
    _, R = k.fk_base(q5); xg = k.R.T @ R[:, 0]
    return np.degrees(np.arctan2(xg[1], xg[0])) + JAW_ANG_OFF


def aligned_roll(p_grid, roll0=-50.0, extra=None, cube_angle=0.0):
    """턱 축이 큐브 면(cube_angle, 격자 x축 기준)과 나란해지는 손목 회전을 반복으로 찾음 (90° 대칭 중 가까운 쪽)"""
    roll = roll0
    for _ in range(4):
        q, _ = k.ik_grid(p_grid, roll_urdf_deg=roll, tip_extra_mm=extra)
        err = (jaw_angle_deg(q) - cube_angle + 45) % 90 - 45        # −45~45: 0이면 정렬
        roll -= err
    return roll, err


CUBE_ANG = float(o["orange"].get("angle_deg", 0.0) or 0.0)
CUBE_ANG_RED = float(o["red"].get("angle_deg", 0.0) or 0.0)
_r0, _e0 = aligned_roll((ox, oy, GRASP_Z), -50.0, None, CUBE_ANG)
print(f"주황 큐브 방향(격자 x축 기준) {CUBE_ANG:.1f}°")
ROLL_CANDIDATES = [_r0, _r0 + 90.0, _r0 - 90.0]   # 턱을 격자 x축 또는 y축과 나란히(큐브 면과 평행). 간섭 검사로 최종 선택
print(f"턱 정렬 손목 회전: {_r0:.1f}° (잔여 각 {_e0:.1f}°), 후보 {np.round(ROLL_CANDIDATES,1)}")


def make_plan(roll):
    # 기준점: 열린 상태는 두 턱 중간점(EXTRA_OPEN), 문 상태는 큐브 중심(EXTRA_HOLD). 닫으면 큐브가 고정 턱 쪽으로 밀려 EXTRA_HOLD 위치에 잡힘.
    plan = [("주황 위 대기", (ox, oy, GRASP_Z + HOVER), GRIP_OPEN, EXTRA_OPEN),
            ("주황 잡기 높이", (ox, oy, GRASP_Z), GRIP_OPEN, EXTRA_OPEN),
            ("닫기", (ox, oy, GRASP_Z), GRIP_CLOSE, EXTRA_OPEN),
            ("들어올리기", (ox, oy, GRASP_Z + HOVER), GRIP_CLOSE, EXTRA_HOLD),
            ("빨강 위 대기", (rx, ry, PLACE_Z + 40.0), GRIP_CLOSE, EXTRA_HOLD),   # 110 mm 이상은 손목 한계에 가까워 40 mm 여유만
            ("놓기 높이", (rx, ry, PLACE_Z), GRIP_CLOSE, EXTRA_HOLD),
            ("열기", (rx, ry, PLACE_Z), GRIP_OPEN, EXTRA_HOLD),
            ("올리기", (rx, ry, PLACE_Z + 40.0), GRIP_OPEN, EXTRA_HOLD)]
    qs = []; q_prev = None; worst = 0.0
    for name, p, g, extra in plan:
        q, e = k.ik_grid(p, roll_urdf_deg=roll, q0_reading=q_prev, tip_extra_mm=extra)
        qs.append(np.r_[q, g]); q_prev = q; worst = max(worst, e)
    return plan, qs, worst


# ---- 손목 회전 선택: 후보마다 계획을 만들고 시뮬에서 빨강 큐브와의 최소 거리를 재서 여유가 가장 큰 것을 고른다 ----
from sim_check import Checker
from wrist_view import open_wrist, wrist_measure
checker = Checker(red_xy=(rx, ry))
Q_START = np.array([0.0, -5.0, 95.0, 49.0, -96.0, GRIP_OPEN])   # 휴식 자세(검사 시작점). 실제 출발은 현재 자세
best = None
print("손목 회전 후보 검사 (빨강 큐브와 최소 거리, 음수 = 관통):")
for roll in ROLL_CANDIDATES:
    plan, qs, worst_ik = make_plan(roll)
    if worst_ik > 3.0:
        print(f"  roll {roll:6.1f}°: IK 잔차 {worst_ik:.1f} mm → 제외"); continue
    if max(abs(q[4]) for q in qs) > 165.0:      # 손목 읽기값이 ±180 경계 근처면 보간이 반대로 350° 돌 수 있음(케이블 감김) → 제외 (09-29 B-8)
        print(f"  roll {roll:6.1f}°: 손목 읽기값 {max(abs(q[4]) for q in qs):.0f}° (±180 경계) → 제외"); continue
    dists = checker.sweep(Q_START, [n for n, *_ in plan], qs)
    margin = min(dd for _, dd in dists)
    step = min(dists, key=lambda x: x[1])[0]
    print(f"  roll {roll:6.1f}°: 최소 여유 {margin:6.1f} mm (단계: {step})  IK 최대 잔차 {worst_ik:.1f} mm")
    # 기본값(−50)은 여유 10 mm 이상이면 우선 채택, 아니면 여유 최대
    score = margin + (5.0 if roll == ROLL_CANDIDATES[0] and margin >= 10.0 else 0.0)   # 정렬 기본값 우선
    if best is None or score > best[0]:
        best = (score, roll, plan, qs, margin)
if best is None:
    sys.exit("모든 손목 회전에서 닿지 않는 위치 → 중단")
_, roll_sel, plan, qs, margin = best
print(f"→ 손목 회전 {roll_sel:.0f}° 선택 (여유 {margin:.1f} mm)" + ("  ⚠ 5 mm 미만: 큐브가 너무 가까움, 실행 전 확인" if margin < 5 else ""))
for (name, p, g, extra), q in zip(plan, qs):
    print(f"  {name:10s} 목표 {np.round(p,0)}  관절 {np.round(q[:5],1)}  그리퍼 {g}")
if DRY:
    sys.exit("dry run 종료")

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_follower_arm", cameras={}, position_p_coefficient=48))
robot.connect(calibrate=False)


def current():
    obs = robot.get_observation(); return np.array([obs[f"{j}.pos"] for j in JOINTS], float)


class Blocked(Exception): pass
STATS = {}
REST = np.array([0.0, -5.0, 95.0, 49.0, -96.0, 35.0])   # 실물 휴식 자세(09-29 라이브 읽기값): 접혀서 받침 위에 얹힘, 토크를 꺼도 거의 안 움직임


def go_rest():
    """놓은 큐브를 건드리지 않고 휴식 자세로: (1) 손목 회전 유지한 채 z 140으로 직진 상승 → (2) 같은 높이로 로봇 쪽 80 mm 물러남
    → (3) 그 자리에서 손목만 기본 회전(−50)으로 → (4) 관절 보간으로 휴식 자세. (09-29 B2: 손목을 돌린 회차에서 올라간 직후
    관절 보간이 큐브 위에서 손목을 되돌리며 쌓은 큐브를 건드림 → 순서 변경)"""
    try:
        q0 = current(); p0 = k.fk_grid(q0[:5]); roll_now = k.reading_to_urdf(q0[:5])[4]
        move_cart((p0[0], p0[1], 140.0), roll_now, None, GRIP_OPEN, "위로 140", target=None)
        p1 = k.fk_grid(current()[:5])
        move_cart((p1[0], max(p1[1] - 80.0, 60.0), 140.0), roll_now, None, GRIP_OPEN, "물러남", target=None)
        q_r, e = k.ik_grid(k.fk_grid(current()[:5]), roll_urdf_deg=-50.0, q0_reading=current()[:5])
        if e < 5.0 and abs(q_r[4] - current()[4]) < 180.0: move_to(np.r_[q_r, GRIP_OPEN], "손목 원위치", settle=False)
    except Exception as ex:
        print("  안전 이동 일부 생략:", ex)
    move_to(REST, "휴식 자세", settle=False)


def move_cart(p_to, roll, extra, grip, label, target=None, step_mm=5.0, speed_mm_s=40.0):
    """격자 좌표에서 손끝 기준점을 직선으로 옮김(경유점마다 IK). 수직 하강·상승용: 관절 보간은 접힌 자세에서 호를 그려 큐브를 스침(09-29 B-3)."""
    q0 = current(); p0 = k.fk_grid(q0[:5], extra); p_to = np.asarray(p_to, float)
    n = max(2, int(np.linalg.norm(p_to - p0) / step_mm) + 1)
    wps = []; qp = q0[:5]
    for i in range(1, n + 1):
        pw = p0 + (p_to - p0) * i / n
        qw, e = k.ik_grid(pw, roll_urdf_deg=roll, q0_reading=qp, tip_extra_mm=extra)
        if e > 3.0: print(f"  ⚠ 직선 경유점 IK 잔차 {e:.1f} mm at {np.round(pw,0)}"); 
        wps.append(np.r_[qw, grip]); qp = qw
    dur = max(1.0, np.linalg.norm(p_to - p0) / speed_mm_s); frames = int(dur * 30)
    seq = [q0] + wps
    for f in range(1, frames + 1):
        a = 0.5 - 0.5 * np.cos(np.pi * f / frames); t = a * (len(seq) - 1); i = min(int(t), len(seq) - 2); u = t - i
        q = seq[i] + (seq[i + 1] - seq[i]) * u
        robot.send_action({j: float(v) for j, v in zip([f"{x}.pos" for x in JOINTS], q)}); time.sleep(1 / 30)
    time.sleep(0.4)
    arr = k.fk_grid(current()[:5], extra); print(f"  [{label}] 직선 이동 도착 FK 격자(기준점) {np.round(arr,1)}")
    if target is not None and arr[2] - target[2] > 8.0:
        print(f"  ⚠ 목표보다 {arr[2]-target[2]:.0f} mm 높은 곳에서 멈춤 → 무언가에 막힘. 정지합니다."); raise Blocked()
    # 정착 보정(2회)
    tgt = p_to.copy(); cmd = tgt.copy(); q_prev = current()
    for it in range(2):
        err = tgt - arr
        if np.linalg.norm(err) < 2.0: break
        cmd = cmd + err; q_c, e_c = k.ik_grid(cmd, roll_urdf_deg=roll, q0_reading=q_prev[:5], tip_extra_mm=extra)
        if e_c > 3.0: break
        q_c6 = np.r_[q_c, grip]; qa = current(); nn = 24
        for i in range(1, nn + 1):
            robot.send_action({j: float(v) for j, v in zip([f"{x}.pos" for x in JOINTS], qa + (q_c6 - qa) * (0.5 - 0.5 * np.cos(np.pi * i / nn)))}); time.sleep(1 / 30)
        time.sleep(0.3); q_prev = current(); arr = k.fk_grid(q_prev[:5], extra)
        print(f"    정착 {it+1}: 오차 {np.round(err,1)} → 재도착 {np.round(arr,1)}")


def wrist_servo(WT, ox, oy, z, roll, extra, max_iter=6):
    """대기 높이에서 손목 카메라로 큐브를 목표 픽셀·각도(잡기 중심 바로 아래일 때의 모습)에 맞춘다.
    방향·감도는 추측하지 않는다: 격자 x, y로 8 mm씩 시험 이동해 '실제 FK 이동량' 대비 픽셀 변화(자코비안)를 재고,
    보정마다 관측된 변화로 감도를 갱신(시컨트)하며 감쇠 0.8로 수렴시킨다. 회전은 한 번 돌려 보고 부호를 정한다.
    반환: (dx, dy, droll) 누적 보정량(격자 mm, 도)."""
    global WRIST
    if WRIST is None:
        WRIST = open_wrist()
    def goto(gx, gy, r):
        q, _ = k.ik_grid((gx, gy, z), roll_urdf_deg=r, q0_reading=current()[:5], tip_extra_mm=extra); move_to(np.r_[q, GRIP_OPEN], "서보", extra, settle=False)
        return k.fk_grid(current()[:5], extra)[:2]          # 실제 도착 (격자 mm)
    def meas(tag):
        m = wrist_measure(WRIST, save=os.path.expanduser(f"~/robotics/calib/pick_{RUN_TAG}_servo_{tag}.png"))
        if m is None: raise RuntimeError("손목 카메라에서 주황 큐브를 못 찾음")
        return m
    def px(m): return np.array([m["u"], m["v"]])
    mm_per_px = 30.0 / WT["w_px"]; tgt = np.array([WT["u"], WT["v"]])
    cx, cy, cr = ox, oy, roll
    a0 = goto(cx, cy, cr); m0 = meas("0")
    print(f"  [손목 서보] 현재 픽셀 ({m0['u']:.0f},{m0['v']:.0f}) 각도 {m0['angle']:.1f}° / 목표 ({WT['u']:.0f},{WT['v']:.0f}) {WT['angle_deg']:.1f}°  ({mm_per_px:.2f} mm/px)")
    # 1) 자코비안: x, y로 8 mm 시험 이동, 실제 FK 이동량으로 나눔
    a1 = goto(cx + 8, cy, cr); m1 = meas("jx"); a2 = goto(cx + 8, cy + 8, cr); m2 = meas("jy")
    dpx1, dmm1 = px(m1) - px(m0), a1 - a0; dpx2, dmm2 = px(m2) - px(m1), a2 - a1
    # 이동이 두 축 모두에 섞일 수 있으므로 2×2 선형 풀이: [dpx1 dpx2] = J [dmm1 dmm2]
    J = np.column_stack([dpx1, dpx2]) @ np.linalg.inv(np.column_stack([dmm1, dmm2]))
    print(f"    시험 이동 실제 {np.round(dmm1,1)}, {np.round(dmm2,1)} mm → 픽셀/mm 자코비안 {np.round(J,2).tolist()}")
    if abs(np.linalg.det(J)) < 0.5: raise RuntimeError("자코비안이 특이(시험 이동이 너무 작음)")
    # 2) 위치 반복 보정 (감쇠 0.8, 시컨트 갱신)
    m = m2; a_prev = a2; cx, cy = cx + 8, cy + 8
    for it in range(max_iter):
        e = tgt - px(m)
        if np.linalg.norm(e) * mm_per_px < 1.5: break
        d = np.clip(0.8 * np.linalg.solve(J, e), -20, 20)
        a_new = goto(cx + d[0], cy + d[1], cr); m_new = meas(f"p{it+1}")
        dmm = a_new - a_prev; dpx = px(m_new) - px(m)
        if np.linalg.norm(dmm) > 2.0:                       # 시컨트: 관측된 (픽셀 변화, 실제 이동)으로 J 갱신
            J = J + np.outer(dpx - J @ dmm, dmm) / (dmm @ dmm)
        cx, cy = cx + d[0], cy + d[1]; m, a_prev = m_new, a_new
        print(f"    위치 {it+1}: 명령 ({d[0]:+.1f},{d[1]:+.1f}) 실제 {np.round(dmm,1)} → 픽셀 오차 {np.linalg.norm(tgt-px(m))*mm_per_px:.1f} mm")
    # 3) 회전: 한 번 돌려 보고 부호 결정
    ea = ((WT["angle_deg"] - m["angle"]) + 45) % 90 - 45
    if abs(ea) > 2.0:
        cr += ea; goto(cx, cy, cr); m2 = meas("r1"); ea2 = ((WT["angle_deg"] - m2["angle"]) + 45) % 90 - 45
        if abs(ea2) > abs(ea) * 0.7:
            cr -= 2 * ea; goto(cx, cy, cr); m2 = meas("r2"); ea2 = ((WT["angle_deg"] - m2["angle"]) + 45) % 90 - 45
        print(f"    회전: 오차 {ea:+.1f}° → 보정 후 {ea2:+.1f}°"); m = m2
        for it in range(2):                                 # 회전 뒤 위치 마무리
            e = tgt - px(m)
            if np.linalg.norm(e) * mm_per_px < 1.5: break
            d = np.clip(0.8 * np.linalg.solve(J, e), -20, 20); goto(cx + d[0], cy + d[1], cr); cx, cy = cx + d[0], cy + d[1]; m = meas(f"pf{it+1}")
    fin = np.linalg.norm(tgt - px(m)) * mm_per_px
    print(f"    최종 픽셀 오차 {fin:.1f} mm, 각도 오차 {((WT['angle_deg']-m['angle'])+45)%90-45:+.1f}°" + ("  ⚠ 3 mm 초과" if fin > 3 else ""))
    STATS["servo_err_mm"] = round(float(fin), 1); STATS["servo_ang"] = round(float(((WT['angle_deg']-m['angle'])+45)%90-45), 1)
    # 실제 도착 위치 기준으로 잡기 목표를 정함(명령-실제 차이 보정)
    a_fin = k.fk_grid(current()[:5], extra)[:2]
    return a_fin[0] - ox, a_fin[1] - oy, cr - roll


def move_to(q_t, label, extra=None, target=None, settle=True):
    q0 = current(); q_t = np.array(q_t, float)
    if abs(q_t[4] - q0[4]) > 180.0:       # 손목 롤 ±180 경계: 최단 경로가 경계를 넘으면 위험 → 중단
        print(f"  ⚠ 손목 회전 명령 {q_t[4]:.0f}° ↔ 현재 {q0[4]:.0f}°: ±180 경계를 넘는 회전. 정지합니다."); raise Blocked()
    dq = np.abs(q_t - q0); dur = max(0.8, dq[:5].max() / MAX_DEG_PER_S); n = int(dur * 30)
    for i in range(1, n + 1):
        a = 0.5 - 0.5 * np.cos(np.pi * i / n)
        robot.send_action({j: float(v) for j, v in zip([f"{x}.pos" for x in JOINTS], q0 + (q_t - q0) * a)})
        time.sleep(1 / 30)
    time.sleep(0.4)
    arr = k.fk_grid(current()[:5], extra); print(f"  [{label}] 도착 FK 격자(기준점) {np.round(arr,1)}")
    if target is not None and arr[2] - target[2] > 8.0:
        print(f"  ⚠ 목표보다 {arr[2]-target[2]:.0f} mm 높은 곳에서 멈춤 → 무언가에 막힘(큐브 위에 얹힘). 정지합니다.")
        raise Blocked()
    # 정착 보정: 중력 처짐·백래시로 명령 위치에 못 미치면 (목표 − 실제)만큼 더 명령 (최대 2회, 2 mm 이내면 종료)
    if target is not None and settle:
        tgt = np.asarray(target, float); cmd = tgt.copy(); q_prev = current()
        for it in range(2):
            err = tgt - arr
            if np.linalg.norm(err) < 2.0: break
            cmd = cmd + err; roll_now = k.reading_to_urdf(q_prev[:5])[4]
            q_c, e_c = k.ik_grid(cmd, roll_urdf_deg=roll_now, q0_reading=q_prev[:5], tip_extra_mm=extra)
            if e_c > 3.0: break
            q_c6 = np.r_[q_c, q_t[5]]; q0 = current(); n = 24
            for i in range(1, n + 1):
                robot.send_action({j: float(v) for j, v in zip([f"{x}.pos" for x in JOINTS], q0 + (q_c6 - q0) * (0.5 - 0.5 * np.cos(np.pi * i / n)))}); time.sleep(1 / 30)
            time.sleep(0.3); q_prev = current(); arr = k.fk_grid(q_prev[:5], extra)
            print(f"    정착 {it+1}: 오차 {np.round(err,1)} → 재도착 {np.round(arr,1)}")


try:
    input("Enter를 누르면 토크를 켜고 시작합니다 (팔 주변을 비우세요) > ")   # 자동 모드에서도 이 확인은 유지
    robot.bus.enable_torque()
    WT = GC.get("wrist_target"); retries = 0
    for i, ((name, p, g, extra), q) in enumerate(zip(plan, qs)):
        ask(f"다음: {name} → Enter ")
        if name in ("주황 잡기 높이", "들어올리기", "놓기 높이", "올리기"):     # 수직 이동은 격자 직선 경로로
            roll_now = k.reading_to_urdf(current()[:5])[4]
            move_cart(p, roll_now, extra, g, name, target=p)
        else:
            move_to(q, name, extra, p, settle=(name != "열기"))   # 열기는 놓은 뒤라 정착 보정 없음. 닫기는 닫는 순간 팔이 내려앉으므로 정착 보정으로 되돌림
        snap(f"{i+1}_{name}")
        # ---- 물림 확인: 들어올린 직후 손목 사진에서 큐브가 가까이(폭 ≥ 200 px)·아래 중앙(v ≥ 300)·기울지 않게(|각| ≤ 12°) 보여야 잡힌 것 ----
        if name == "들어올리기" and WT and not DRY:
            m = wrist_measure(WRIST, save=os.path.expanduser(f"~/robotics/calib/pick_{RUN_TAG}_grip_check{retries}.png"))
            held = m is not None and m["w_px"] >= 200 and m["v"] >= 300 and abs(m["angle"]) <= 12
            print(f"  [물림 확인] " + ("없음" if m is None else f"폭 {m['w_px']:.0f} px, v {m['v']:.0f}, 각 {m['angle']:.1f}°") + (" → 잡힘" if held else " → 못 잡음"))
            STATS["grip_ok"] = int(bool(held))
            if not held and retries < 1:
                retries += 1; print("  다시 잡기: 열고 서보부터 반복")
                move_to(np.r_[current()[:5], GRIP_OPEN], "열기(재시도)", extra, settle=False)
                dx, dy, droll = wrist_servo(WT, plan[0][1][0], plan[0][1][1], GRASP_Z + HOVER, roll_sel, EXTRA_OPEN)
                ox2, oy2, roll2 = plan[0][1][0] + dx, plan[0][1][1] + dy, roll_sel + droll
                q_prev = current()[:5]
                for j in range(1, 4):
                    nm, pp, gg, ex = plan[j]; pp2 = (ox2, oy2, pp[2])
                    qj, ej = k.ik_grid(pp2, roll_urdf_deg=roll2, q0_reading=q_prev, tip_extra_mm=ex); qs[j] = np.r_[qj, gg]; plan[j] = (nm, pp2, gg, ex); q_prev = qj
                for j in range(1, 4):
                    nm, pp, gg, ex = plan[j]
                    if nm == "닫기": move_to(qs[j], nm, ex, pp, settle=True)
                    else: move_cart(pp, k.reading_to_urdf(current()[:5])[4], ex, gg, nm, target=pp)
                    snap(f"{j+1}_{nm}_retry")
                m = wrist_measure(WRIST, save=os.path.expanduser(f"~/robotics/calib/pick_{RUN_TAG}_grip_check{retries}.png"))
                held = m is not None and m["w_px"] >= 200 and m["v"] >= 300 and abs(m["angle"]) <= 12
                print(f"  [물림 재확인] " + ("없음" if m is None else f"폭 {m['w_px']:.0f} px, v {m['v']:.0f}, 각 {m['angle']:.1f}°") + (" → 잡힘" if held else " → 못 잡음, 그대로 진행"))
                STATS["grip_ok"] = int(bool(held)); STATS["grip_retry"] = 1
        if i == 0 and WT:
            dx, dy, droll = wrist_servo(WT, ox, oy, GRASP_Z + HOVER, roll_sel, EXTRA_OPEN)
            ox2, oy2, roll2 = ox + dx, oy + dy, roll_sel + droll
            print(f"  서보 결과: 큐브 위치 보정 ({dx:+.1f}, {dy:+.1f}) mm, 손목 회전 보정 {droll:+.1f}° → 잡기 단계 재계산")
            STATS.update(dx=round(dx, 1), dy=round(dy, 1), droll=round(droll, 1))
            # 서보가 찾은 보정(계산 기준점 → 실제 잡기 중심)은 손에 든 큐브에도 그대로 적용되므로 놓기 단계에도 같은 (dx,dy)를 더한다.
            # 놓을 때 손목 회전은 빨강 큐브 방향에 맞춤: 잡은 큐브는 턱과 나란하므로 (빨강 각도 − 주황 각도)만큼 더 돌린다.
            roll_place = roll2 + (((CUBE_ANG_RED - CUBE_ANG) + 45) % 90 - 45)
            print(f"    놓기 손목 회전 {roll_place:.1f}° (빨강 방향 {CUBE_ANG_RED:.1f}°, 주황 {CUBE_ANG:.1f}°)")
            # 서보 보정(계산 기준점 → 실제 잡기 중심)은 그리퍼에 붙은 오프셋이므로 그리퍼 프레임으로 옮긴 뒤,
            # 놓기 자세(팔 방향·손목 회전이 다름)에서 다시 격자 좌표로 되돌려 적용한다. (B-1 좌우 바꿈 실패의 원인: 격자 좌표 그대로 더했음)
            _, R_grasp = k.fk_base(current()[:5]); d_grip = R_grasp.T @ (k.R @ np.array([dx, dy, 0.0]))
            q_prev = current()[:5]
            for j in range(1, len(plan)):
                nm, pp, gg, ex = plan[j]
                if j <= 3:
                    pp2 = (ox2, oy2, pp[2]); rj = roll2
                else:
                    q0, _ = k.ik_grid((rx, ry, pp[2]), roll_urdf_deg=roll_place, q0_reading=q_prev, tip_extra_mm=ex)
                    _, R_pl = k.fk_base(q0); d = k.R.T @ (R_pl @ d_grip)
                    pp2 = (rx + d[0], ry + d[1], pp[2]); rj = roll_place
                qj, ej = k.ik_grid(pp2, roll_urdf_deg=rj, q0_reading=q_prev, tip_extra_mm=ex); qs[j] = np.r_[qj, gg]; plan[j] = (nm, pp2, gg, ex); q_prev = qj
                print(f"    {nm:8s} → {np.round(pp2,0)}  IK 잔차 {ej:.1f} mm")
            print(f"    (그리퍼 기준 잡기 중심 오프셋 {np.round(d_grip[:2],1)} mm → 놓기 자세에서 격자 보정 {np.round(d[:2],1)} mm; 잡기 자세에서는 ({dx:+.1f},{dy:+.1f}))")
    ask("완료. Enter를 누르면 휴식 자세(접힌 자세)로 돌아간 뒤 토크를 끄고 종료합니다 > ")
    STATS["finished"] = True
    go_rest()
except KeyboardInterrupt:
    print("\n중단: 현재 위치 정지")
except Blocked:
    snap("blocked"); print("막힘으로 중단. 사진 저장. 휴식 자세로 돌아갑니다.")
    try: move_to(np.r_[current()[:5], GRIP_OPEN], "열기", settle=False); go_rest()
    except Exception as ex: print("  휴식 자세 이동 실패:", ex)
finally:
    robot.disconnect(); print("연결 해제")
    if CAP is not None: CAP.release()
    if WRIST is not None: WRIST.release()
    # ---- 결과 기록 (성공/실패 + 이유 한 줄) ----
    try:
        ans = input("결과 입력: s=성공(주황이 빨강 위에 3초 유지) / f=실패, 뒤에 이유 (예: 'f 놓을 때 모서리 걸려 떨어짐') / 빈칸=기록 안 함 > ").strip()
    except EOFError:
        ans = ""
    if ans:
        ok = ans[0].lower() == "s"; reason = ans[1:].strip()
        hdr = "time,tag,ox,oy,rx,ry,cube_ang,roll,servo_err_mm,servo_ang,dx,dy,droll,finished,success,reason,grip_ok,grip_retry,P\n"
        row = f"{RUN_TAG},{TAG},{ox:.1f},{oy:.1f},{rx:.1f},{ry:.1f},{CUBE_ANG:.1f},{roll_sel:.1f},{STATS.get('servo_err_mm','')},{STATS.get('servo_ang','')},{STATS.get('dx','')},{STATS.get('dy','')},{STATS.get('droll','')},{int(STATS.get('finished', False))},{int(ok)},\"{reason}\",{STATS.get('grip_ok','')},{STATS.get('grip_retry',0)},48\n"
        new = not os.path.exists(RUN_LOG)
        with open(RUN_LOG, "a") as f:
            if new: f.write(hdr)
            f.write(row)
        rows = open(RUN_LOG).read().strip().splitlines()[1:]
        same = [r for r in rows if r.split(",")[1] == TAG]; n_ok = sum(int(r.split(",")[14]) for r in same)
        print(f"기록: {RUN_LOG}  (조건 {TAG or '-'}: {n_ok}/{len(same)} 성공)")
