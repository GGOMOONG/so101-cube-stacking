# 2분 요약 영상 구성안

목표: 심사자가 2분 안에 "무엇을, 어떻게, 결과가 무엇인지"를 본다. 말은 없고 자막만. 배경음악은 조용한 것 하나. 1080p 가로.

**재료 위치**
- 학습 정책 실행 영상: `~/.cache/huggingface/lerobot/moonbeards/<데이터셋>/videos/observation.images.top/chunk-000/file-000.mp4` (손목은 `observation.images.wrist`). 에피소드 여러 개가 한 파일에 이어져 있으니 원하는 구간을 잘라 쓴다.
- 루프 시연 휴대폰 영상: `~/robotics/videos/loop_stack_unstack1.MP4`(성공), `loop_stack_unstack2.MP4`(실패)
- 환경 사진: `~/robotics/photos/`
- 보정·시뮬 그림: `~/robotics/calib/top_homography_check.png`, `calib/cube_01_located.png`, `calib/grasp_center_wrist.png`, `figures/sim/`, `figures/fig_pipelines_ko.png`, `figures/fig_results_ko.png`
- 고전 파이프라인 실행 사진: `~/robotics/calib/pick_0929_042006_*_top.png`(8단계), `*_wrist.png`
- **고전 파이프라인 실행 영상은 아직 없음** → 아래 "촬영할 것" 참고

## 장면 구성 (총 2:00)

### 장면 1 · 0:00–0:08

**내용** 제목. 환경 전체 사진 위에 텍스트

**재료** `photos/setup_full1.JPG`

**자막** SO-101 색 큐브 쌓기: 모방학습 대 고전 파이프라인

**Caption** Learned vs. classical manipulation on a low-cost arm

### 장면 2 · 0:08–0:18

**내용** 문제. 탑 카메라 시점에서 큐브 두 개, 팔이 주황을 빨강 위에 올리는 장면 (v4 성공 회차)

**재료** v4 쌓기 평가 `..._v4_stack_eval_v1_20260921_180005` 성공 에피소드 1개, 2배속

**자막** 과제: 주황 큐브를 빨강 위에 쌓기. 성공 = 3초 유지

**Caption** Task: stack orange on red, hold 3 s

### 장면 3 · 0:18–0:33

**내용** 모방학습 과정. 시연(리더 암 손 포함) → 첫 실패(v1) → DAgger 교정 → v4

**재료** 시연: `color_cube_stack_unstack2` 첫 에피소드 3초 / v1 실패: `..._stack2_eval_v1_20260916_020659` 3초 / DAgger: `..._dagger_..._stack2_20260921_015414` 사람이 이어받는 순간 4초 / v4: 2번과 다른 성공 회차 3초

**자막** 시연 176회 → ACT 1/10 → SmolVLA + DAgger 40회 → 9/10

**Caption** 176 demos → ACT 1/10 → SmolVLA + 40 DAgger corrections → 9/10

### 장면 4 · 0:33–0:40

**내용** 언어 조건. 같은 모델이 문장에 따라 쌓기/허물기, 반사실에서는 정지

**재료** 루프 영상 `loop_stack_unstack1.MP4` 앞 20초를 3배속 + 반사실 `..._v4_counterfactual_...` 2초

**자막** 한 모델, 두 문장: "쌓아라"

**Caption** "허물어라". 이미 쌓여 있으면 가만히 (5/5) / One model, two sentences; no action when already stacked

### 장면 5 · 0:40–0:48

**내용** 학습의 한계. 루프 실패 장면(큐브가 옆에 떨어진 뒤 회복 못 함)

**재료** `loop_stack_unstack2.MP4` 실패 구간 8초

**자막** 시연 범위를 벗어나면 회복하지 못한다

**Caption** Cannot recover outside the demonstration range

### 장면 6 · 0:48–1:05

**내용** 고전 파이프라인 구축. 체스판 → 격자 검출 그림 → 큐브 검출 그림 → 4점 짚기 사진 → MuJoCo 궤적

**재료** `calib/top_homography_check.png`(2초) → `calib/cube_01_located.png`(2초) → 짚기 장면(촬영 필요, 없으면 `photos/cam_wrist1.JPG`) → `figures/sim/plan_*.png` 8장을 0.5초씩

**자막** 학습 없이: 카메라 보정 0.45 mm → 큐브 위치 ≤3 mm → 관절 보정 1.5 mm → 역기구학 → MuJoCo 검증

**Caption** No learning: calibration → cube position → kinematics → MuJoCo

### 장면 7 · 1:05–1:20

**내용** 고전 파이프라인 실행. 손목 카메라 서보가 큐브를 가운데로 맞추고 집어 올려 쌓는 장면

**재료** **촬영 필요**: 휴대폰으로 A 배치 1회 전체(약 70초)를 찍어 3배속. 손목 카메라 화면은 `calib/pick_0929_042006_*_wrist.png` 8장을 끼워 넣기

**자막** 손목 카메라 시각 서보 0.4~1.5 mm → 잡기 20/20

**Caption** Wrist-camera visual servo → grasp 20/20

### 장면 8 · 1:20–1:32

**내용** 범위 밖 배치. 고전은 좌우를 바꿔도 되고, v4는 인식 못 함

**재료** **촬영 필요**: 고전 B-1(좌우 바꿈) 1회 3배속 + v4 B-1 실패 장면(촬영 또는 없으면 자막만)

**자막** 좌우를 바꾸면: 고전 7/10, 학습 0/10

**Caption** Mirrored layout: classical 7/10, learned 0/10

### 장면 9 · 1:32–1:48

**내용** 결과 그래프

**재료** `figures/fig_results_ko.png` (막대가 하나씩 올라오게)

**자막** 시연 범위 안 10/10 대 4/10, 범위 밖 7/10 대 0/10

**Caption** In-distribution 10/10 vs 4/10, out 7/10 vs 0/10

### 장면 10 · 1:48–2:00

**내용** 결론 + 다음. 두 파이프라인 도식 위에 텍스트

**재료** `figures/fig_pipelines_ko.png`

**자막** 두 방식은 다른 곳에서 무너진다. 다음: 놓기 서보, 혼합 제어, 심투리얼

**Caption** They fail in different places. Next: closed-loop placing, hybrid control, sim-to-real


**촬영할 것 (격자 종이를 테이프 자리에 되돌린 뒤, 휴대폰 가로, 팔 전체와 큐브가 보이는 각도)**
1. 고전 A 배치 1회 전체: `python pick_stack.py --auto --tag V` (장면 7)
2. 고전 B-1 좌우 바꿈 1회: 같은 명령 (장면 8)
3. 선택: 4점 짚기 장면 10초 (`python touch_points.py`, 실제로 기록하지 않아도 됨) (장면 6)
4. 선택: v4 B-1 실패 1회 (격자 걷고 `P=16 zsh v4_eval.sh V`) (장면 8)

**편집 요령**
- 실행 장면은 전부 2~3배속. 대기·Enter 구간은 잘라낸다.
- 자막은 화면 아래 한 줄, 한국어 위·영문 아래 또는 제출처 언어 하나만.
- 숫자가 나오는 자막은 최소 2초 유지.
- 마지막 프레임에 저장소 주소와 이름.

### 장면 11 · 추가 (10-09) — 시뮬레이터

**내용** MuJoCo 시뮬 안에서 시뮬 전용 정책(v2 10k)이 쌓기를 완수하는 장면(탑·손목·외부 뷰 + HUD)

**재료** `videos/sim_policy_vid_sim2_10k_A_3.mp4` (B 평가 뒤 녹화 예정), 3배속 15초

**자막** MuJoCo 시뮬레이터: 실측 동역학·카메라, 자동 생성 시연 296회로 학습, 사람 시연 0회

**Caption** MuJoCo: measured dynamics and cameras, 296 auto-generated demos, zero human demos

### 장면 12 · 추가 — 시뮬 정책을 실물에

**내용** 같은 정책이 실물 팔에서 쌓기를 완수하는 장면

**재료** **촬영 필요**: `videos/sim2_real_success.MP4` (휴대폰, 성공 회차 1회). 없으면 장면 생략(스크립트가 자동 건너뜀)

**자막** 같은 정책을 실물 팔에: 실물 시연 없이 쌓기 완수 (2/10)

**Caption** The same policy on the real arm: stacks without any real demonstration (2/10)

### 장면 13 · 추가 — 시뮬→실물 결과

**내용** 학습 곡선 그림(6초) → 실물 비교 막대그래프(8초) → 도식 위 "다음"(5초)

**재료** `figures/fig_sim_curves_ko.png`, `figures/fig_sim2real_ko.png`, `figures/fig_pipelines_ko.png`

**자막** 시뮬 안 천장 3~4/10, 더 학습하면 떨어진다 / 실물: 시뮬 전용 2/10 대 실물 시연 정책 3~4/10, 시뮬 내 성공률은 실물 전이를 예측하지 못함(v3b 0/20) / 다음: 시뮬 시연 다양화, 놓기 서보, 혼합 제어

**Caption** Ceiling 3–4/10 in sim; more training hurts / Real: sim-only 2/10 vs real-demo 3–4/10; sim success did not predict transfer / Next: diversify sim demos, closed-loop placing, hybrid control

**길이** 장면 11~13 추가로 총 약 2:35. 2분을 지키려면 장면 6(루프)·8(루프 실패)을 각 8초씩 줄인다.

**촬영할 것 (추가)** 시뮬 정책(v2 10k) 실물 성공 회차 1회 휴대폰 촬영 → `videos/sim2_real_success.MP4`
