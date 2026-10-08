#!/usr/bin/env python3
"""SO-101 정기구학을 URDF의 관절 원점·축만 읽어 numpy로 직접 계산한다 (placo 없이).
각 관절: 부모 링크 → 고정 변환(origin xyz, rpy) → 축(axis) 둘레 회전(각도) → 자식 링크.
사용: from so101_fk import SO101FK; fk = SO101FK(); T = fk.fk_deg([pan, lift, elbow, wflex, wroll, grip])
"""
import xml.etree.ElementTree as ET
import numpy as np

URDF = "/Users/m4pro/robotics/so101_model/so101_new_calib.urdf"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF 관례: R = Rz(yaw) Ry(pitch) Rx(roll)


def axis_angle(axis, th):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def T_of(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


class SO101FK:
    def __init__(self, urdf=URDF, tip="gripper_frame_link"):
        root = ET.parse(urdf).getroot()
        joints = {}
        for j in root.findall("joint"):
            o = j.find("origin"); a = j.find("axis")
            xyz = np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in o.get("rpy").split()]) if o is not None and o.get("rpy") else np.zeros(3)
            joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                         child=j.find("child").get("link"), xyz=xyz, rpy=rpy,
                                         axis=np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([0, 0, 1.0]))
        # 베이스에서 tip까지의 사슬을 child→parent로 거슬러 올라가 구성
        by_child = {v["child"]: (k, v) for k, v in joints.items()}
        chain = []
        link = tip
        while link in by_child:
            name, j = by_child[link]
            chain.append((name, j)); link = j["parent"]
        self.chain = chain[::-1]
        self.base = link
        self.joints = joints

    def fk(self, q_rad: dict):
        T = np.eye(4)
        for name, j in self.chain:
            T = T @ T_of(rpy_to_R(*j["rpy"]), j["xyz"])
            if j["type"] == "revolute":
                T = T @ T_of(axis_angle(j["axis"], q_rad.get(name, 0.0)), np.zeros(3))
        return T

    def fk_deg(self, q_deg):
        q = {n: np.deg2rad(v) for n, v in zip(JOINTS, q_deg)}
        return self.fk(q)


if __name__ == "__main__":
    fk = SO101FK()
    print("chain:", [n for n, _ in fk.chain], "base:", fk.base)
    for n, j in fk.chain:
        print(f"  {n:20s} axis={j['axis']}  xyz(mm)={np.round(j['xyz']*1000,1)}  rpy={np.round(j['rpy'],3)}")
    for name, q in [("zero", [0, 0, 0, 0, 0, 0]), ("lift+45", [0, 45, 0, 0, 0, 0]), ("pan+90", [90, 0, 0, 0, 0, 0]), ("elbow+90", [0, 0, 90, 0, 0, 0])]:
        T = fk.fk_deg(q); print(f"FK {name:9s}: x={T[0,3]*1000:7.1f} y={T[1,3]*1000:7.1f} z={T[2,3]*1000:7.1f} mm")
