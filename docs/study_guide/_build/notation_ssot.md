# 전역 기호 및 기준 수치 사전 (Notation & Verified Metrics SSOT)

> 모든 챕터(`ch01.md` ~ `ch07.md`, `appendix.md`)는 아래 기호 표기, 단위, 검증된 실측 수치를 **단일 진실 공급원(SSOT)**으로 준수합니다.

## 1. 전역 기호 및 단위 사전

| 기호 | 의미 | 단위 | 비고 |
| :--- | :--- | :---: | :--- |
| $r, \theta, z$ | 웨이퍼·ESC 원통 좌표 | $\text{m}$ (표기 $\text{mm}$), $\text{deg}$ | $R_w = 150\text{ mm}, R_{\text{esc}} = 155\text{ mm}, R_{\text{base}} = 160\text{ mm}$ |
| $t_w$ | 실리콘 웨이퍼 두께 | $\mu\text{m}$ | $t_w = 775\ \mu\text{m}$ |
| $T_w, T_{\text{1D}}, T_{\text{ESC}}$ | 웨이퍼 / 1D·2D 평균 / ESC 상면 온도 | $^\circ\text{C}$ (내부 $\text{K}$) | 목표 온도 $T_{\text{target}} = 40.0^\circ\text{C}$ |
| $Q_1 \sim Q_{10}$ | 10존 매립 히터 전력 | $\text{W}$ | Center Z1, Mid Z2, 외곽 45° 8섹터 Z3~Z10 ($Q_i \ge 0$) |
| $P_{\text{in}}, P_{\text{out}}$ | 내측·외측 백사이드 He 압력 | $\text{Torr}$ | 장비 허용범위 $[3.0, 20.0]\text{ Torr}$ |
| $\beta_{\text{leak}}, \theta_{\text{leak}}, r_{\text{leak}}$ | 에지 He 누설 잔존율 / 방위각 / 시작 반경 | $-, ^\circ, \text{mm}$ | $P_{\text{edge}} = \beta_{\text{leak}} P_{\text{out}}$, $\beta \in [0.05, 1.0]$, $r_{\text{leak}} \le 148\text{ mm}$ |
| $d, \lambda, M(P)$ | 물리 간극 / 평균자유행로 / 온도 점프 거리 | $\mu\text{m}$ | $d = 10\ \mu\text{m}$, $M(P) \propto \lambda \propto 1/P$ |
| $h_{\text{eff}}, h_{\text{solid}}$ | 유효 계면 열전달계수 / 메사 고체 접촉 | $\text{W/m}^2\text{K}$ | $h_{\text{eff}} = h_{\text{solid}} + k_{\text{He}}/(d + M)$, $h_{\text{solid}} = 50$ |
| $k_{\text{Si}}(T), k_{\text{cer}}$ | 실리콘 / 세라믹 열전도도 | $\text{W/m}\cdot\text{K}$ | $k_{\text{Si}} = 148(300/T)^{1.3}$, $\text{Al}_2\text{O}_3 = 30$, $\text{AlN} = 160$ |
| $\text{Bi}_z$ | 웨이퍼 두께 방향 Biot 수 | $-$ | $\text{Bi}_z = h_{\text{eff}} t_w / k_{\text{Si}} \approx 0.009 \ll 0.1$ |
| $Q_{\text{plasma}}, \alpha_{\text{peak}}$ | 플라즈마 총 입열 / 반경 피킹 계수 | $\text{W}, -$ | 기준 $500\text{ W}$, 양수=센터 피킹, 음수=에지 피킹 |
| $S_0, S_{\text{probe}}$ | 1차 적합 / 2-운전점 차분 Pre-Refit 정규화 오차 점수 | $-$ | 노이즈 크기로 나눈 잔차 제곱합 (잠금 문턱 `107.154`, `170.972`) |
| $H_{\text{block}}$ | 활성 장비 정지 인터락 신호 | $\{0, 1\}$ | OEM He Flow High($\ge 20\text{ sccm}$) 또는 Readback 누락 시 `1` |
| $\delta_T, \delta_\nabla$ | 1D/2D–3D FEM 상면 보수 오차 마진 | $^\circ\text{C}, ^\circ\text{C/mm}$ | 사전 승인식 4대 규격에만 적용 (사후 FEM 채점 미포함) |
| $\overline{\rho}(\mathbf{u}^*)$ | 강건 최악 제약 위반도 추정 상계 | $-$ | $\overline{\rho} \le 0$: 사전 승인 가능, $\overline{\rho} > 0$: 보류(`Hold`) |
| $A_{\text{pre}}, V$ | 사전 승인 여부 / 사후 독립 3D FEM 위반 여부 | $\{0, 1\}$ | $n_A = \sum A_{\text{pre}}$ |

## 2. 검증 완료 기준 수치 (Verified Benchmark SSOT)

* **Day 1 (`day1_3d_fem_benchmark.json`):** `medium` 3D FEM 20,956 DOFs (119,765 요소), 전역 열수지 `5.68e-12 W`, 225° 누설 스파이크 **`+5.85°C`**, 수동 에지 보정 시 `dT` 악화 **`5.47°C → 6.58°C`**.
* **Step 1 (`step1_3d_hybrid_verify.json`):** Mode A ROM `M=82` + 10존 히터(`13.88 s` 추출), 2D Polar FVM `24×36 = 864`셀, 3D FEM 대비 MAE **`0.041°C`**(정상) / **`0.063°C`**(225° 누설), 웨이퍼 열수지 **`1.14e-13 W`**, 독립 ESC→칠러 열수지 **`≤ 9.09e-13 W`**, Adjoint 구현 Gradcheck 최대 상대오차 **`1.02e-9`**, 3D FEM FD 감도 오차 `Q_1 0.46%`, `Q_7 0.77%`, `P_out 1.56%`, `beta_leak 8.19%`.
* **Week 2 쇼케이스 (`week2_3d_closed_loop_eval.json`, `test_08_leak_225_show`):** 225° 누설(`β=0.70`), 추정 $\hat{\beta}=0.6958, \hat{\theta}=225.15^\circ$, 대리 모델 예측 `dT=1.13°C` vs 독립 3D FEM 재해석 **`dT=1.14°C`**(`3.64→1.14°C`, **-68.6%**), $3\sigma$ **`2.64→0.73°C`**(**-72.2%**), 평균 **`38.95→40.05°C`**; `test_15_ood_warpage` 단일점 **`S_0=77.3`** 통과 → Pre-Refit **`S_probe=479.3`** 단독 차단.
* **Phase 1.5 (`phase15_verification_suite.json`):**
  * 동일 워밍업 1-스텝 실측(`medium`): 3D FEM **`1,228.2 ms`**, 대리 모델 Forward **`45.63 ms`** (**26.9×**), **Forward+12D Adjoint Backward `57.29 ms`** (**514.5×** vs 12D 중앙유한차분).
  * 대규모 독립 87건 배치: **`n_A = 68`**, 위반 `0/68` (**`0.00%`**), 단측 95% Clopper–Pearson 상한 **`0.0431` (< 5%)**, `H_block` `4/4`, OOD `6/6`, 극심 누설 `5/5` 전부 **100% 차단**.
  * 63pt Operable Envelope: 22/63 복구 가능, 최소 복구 $\beta_{\text{leak}} = 0.70$ 이산 (**`0.6819`** 보간, $\alpha_{\text{peak}} \in [0, 0.30]$).
