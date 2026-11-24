# 플라즈마 식각 챔버 Differentiable Physics / Hybrid SciML 기반 이상치 진단·역설계 파이프라인 설계안 (v1.7)

---

## 1. 목표, 운영 시나리오 및 명시적 설계 가정

* **기술 포지셔닝:** **Differentiable Physics / Hybrid SciML** — 선형 2D ESC 열응답 모델과 미분 가능한 비선형 1D 축대칭 웨이퍼 FVM 레이어를 양방향 결합(Two-way Coupled)하여 이상치 진단과 안전 제약 레시피 역설계를 수행하는 프레임워크.
* **운영 시나리오:** PM 직후 또는 주기적 Chamber Qualification 시 오프라인 계측 웨이퍼(9점 반경 온도 계측)와 장비 외측 He 유량 로그를 활용한 **오프라인 원인 진단 및 레시피 오토튜닝(APC Recipe Re-centering)**.
* **명시적 설계 가정 및 적용 범위 (Scope & Assumptions):**
  1. 장시간 HARC 식각의 정상상태 평탄 구간을 대표 운전점으로 가정하며, 실제 장비 적용 시 레시피 단계별 $dT/dt$, He 압력 및 칠러 이력으로 정상상태 성립을 사전 확인한다.
  2. 계측 모델은 9점 반경 TC 계측($\sigma_T = 0.2^\circ\text{C}$ + 공통 교정 편의 고려)의 시뮬레이션 프록시이며, 실제 제품 웨이퍼와의 열접촉 차이 및 식각 CD/깊이 검증은 실장비 검증 과제로 구분한다.
  3. **구동기·신호 건전성 전제:** 4존 히터 출력($Q_i^{\text{actual}} = \eta_i Q_i^{\text{cmd}}$)과 배면 He 압력 추종은 장비 Readback 및 활성 인터락 신호($H_{\text{block}}$)로 사전 확인됨을 전제한다. Readback 없이 기준·탐색점에서 미구동된 제어축을 새로 사용하는 경우에는 해당 축의 탐색 또는 $\eta_i$ 불확실성 점검을 통과하지 못하면 자동 승인을 보류(`Hold`)한다.
  4. §4.2의 구배·열유속 밀도 제한값($1.5\ ^\circ\text{C/mm}$, $2.0\text{ W/cm}^2$, $0.3\ ^\circ\text{C/mm}$)은 장비별 열응력·CD 검증 전까지 사용하는 보수적 설계 가정치이다.

---

## 2. 기준 2D FEM 해석기(`esc_fem_solver.py`) 물리 및 온도 정의 명세

1. **웨이퍼($R_w = 150\text{ mm}, t_w = 775\ \mu\text{m}$):** 비선형 $k_{\text{Si}}(T) = 148(300/T)^{1.3}$, 면적 정규화 플라즈마 입열 $q''_{\text{plasma}}(r) = \frac{Q_{\text{plasma}}}{\pi R_w^2}[1 + \alpha_{\text{peak}}(1 - 2(r/R_w)^2)]$ ($\int q''_{\text{plasma}}\,dA \equiv Q_{\text{plasma}}$).
   * **FEM vs FVM 온도 물리량 구분:** 2D FEM은 간극 열유속에 **웨이퍼 하면 온도($T_{w,\text{bot}}$)**, 상면 복사 및 사후 제품 품질 채점에 **웨이퍼 상면 온도($T_{w,\text{top}}$)**를 사용한다. 1D FVM은 두께 평균 온도 $T_{\text{1D}}(r)$로 근사한다.
   * **사전 승인용 보수 오차 마진 ($\delta_T, \delta_\nabla$):** Week 1에서 사전 지정한 유한 검증 운전영역(표본 격자)에 대해 점별 온도차 $\|T_{w,\text{top}}^{\text{FEM}} - T_{\text{1D}}\|_\infty$와 구배차 $\|\partial_r(T_{w,\text{top}}^{\text{FEM}} - T_{\text{1D}})\|_\infty$를 실측하고, 안전계수(예: $1.5\times$)를 적용한 보수 마진 $\delta_T, \delta_\nabla$를 사전 승인식에만 적용한다(유한 표본 관측치는 연속 영역 전역 상계의 하계임을 명시하며, 사후 채점 $V$는 마진을 더하지 않은 2D FEM 상면 원규격으로 채점한다).
2. **희박 He 미세 간극($d = 10\ \mu\text{m}$):** Kennard-Yovanovich $h_{\text{eff}}(P(r), T_g(r)) = h_{\text{solid}} + \frac{k_{\text{He}}}{d + M(P(r), T_g(r))}$, $k_r \approx 0$.
   * 웨이퍼→ESC 방향을 양(+)으로 정의: 2D FEM은 $q''_{\text{gap}} = h_{\text{eff}}(T_{w,\text{bot}} - T_{\text{ESC,top}})$, $T_g = \frac{1}{2}(T_{w,\text{bot}} + T_{\text{ESC,top}})$; 1D FVM은 $T_{w,\text{bot}}$ 대신 $T_{\text{1D}}$를 사용한다.
3. **ESC 세라믹($R_{\text{esc}}=155\text{ mm}, t_{\text{cer}}=2.0\text{ mm}$) + Al 베이스($R_{\text{base}}=160\text{ mm}, t_{\text{base}}=15.0\text{ mm}$):**
   * 상수 물성($k_{\text{cer}}, k_{\text{Al}}$), 세라믹–베이스 완전 열접촉, 노출 상면($[150, 155]$, $[155, 160]\text{ mm}$)·외벽 단열(자연 Neumann BC), 하부 전체($[0, 160]\text{ mm}$) 선형 Robin BC($q''_{\text{cool}} = h_{\text{chiller}}(T_b - T_{\text{chiller}})$).
   * 4개 동심원 매립 히터($[0, 50], [50, 100], [100, 135], [135, 150]\text{ mm}$, 면적비 $1 : 3.00 : 3.29 : 1.71$) 균일 체적 발열($Q_i/V_i$). 계면 하중 $q''_{\text{gap}}$가 주어질 때 **ESC 도메인은 100% 선형 2D 열전도 문제**이다.

---

## 3. 양방향 결합 미분 가능 물리 아키텍처

### 3.1 Compact-Support $C^2$ 압력 모델 및 장비 허용 영역
$[0, 1]$ 구간 밖에서 정확히 $0, 1$을 갖는 **$C^2$ Quintic Hermite Smoothstep** $h(\xi) = 6\xi^5 - 15\xi^4 + 10\xi^3$을 FEM과 FVM에 동일 적용한다:
$$s(r) = h\!\left(\frac{r - (r_{\text{split}} - w_s/2)}{w_s}\right), \qquad \ell(r) = h\!\left(\frac{r - r_{\text{leak}}}{w_s}\right), \qquad w_s = 2.0\text{ mm}$$
$$P(r) = \big(1 - \ell(r)\big)\big[(1 - s(r))P_{\text{in}} + s(r)P_{\text{out}}\big] + \ell(r)\,\beta_{\text{leak}}\,P_{\text{out}}, \quad \beta_{\text{leak}} \in [0.05, 1.0]$$
* **누설 위치 경계조건 명시:** 생성기와 추정기 모두 $r_{\text{leak}} + w_s \le R_w$ (즉, $w_s = 2.0\text{ mm}$ 기준 **$r_{\text{leak}} \in [130, 148]\text{ mm}$**)으로 제한하여 $\ell(R_w) \equiv 1$ 및 $P(R_w) \equiv \beta_{\text{leak}} P_{\text{out}}$를 항등 보장한다.
* **장비 허용 영역 ($\mathcal{U}_{\text{HW}}$):** 기준점 $\mathbf{u}_0$, 탐색점 $\mathbf{u}_{\text{probe}}$, 후보 $\mathbf{u}^*$ 모두 아래 영역에 구속되어 $P(r) > 0$가 항등 성립한다:
  $$\mathcal{U}_{\text{HW}}:\quad Q_i \in [0, 400]\text{ W}, \qquad P_k \in [P_{k,\min}^{\text{HW}}, P_{k,\max}^{\text{HW}}] = [3.0, 20.0]\text{ Torr}\ \ (k \in \{\text{in}, \text{out}\})$$

### 3.2 ESC 2D 열응답 모듈 & 4조합 독립 수렴 검증
* **Mode A — 단위하중 FEM 참조해 (`Unit-Load Linear ESC Reference`):**
  * 4개 단위 히터 + $M$개 비등간격 hat 기저($r=100\text{ mm}$, $r \in [130, 150]\text{ mm}$ 집중 배치)에 대한 FEM 경계하중 질의로 ESC 상면 온도 응답($\mathbf{g}_i, \mathbf{G}_q$, 보호 영역 전체 $r \in [0, 155]\text{ mm}$)과 **독립적으로 적분된 하부 칠러 Robin 제열 응답**($c_{i,\text{cool}}, \mathbf{c}_{q,\text{cool}}$, 베이스 하부 전체 $r \in [0, 160]\text{ mm}$)을 별도 추출한다.
  * 원통좌표 면적 가중 $L^2$ 투영으로 계면 열유속을 사상하며, **$Q_{\text{cool}}$을 독립 추출된 Robin 응답으로 계산해** 계면 유입열+히터 발열 합과 대조함으로써 순환 검증을 배제한다.
  * **4조합 독립 수렴 시험:** $(M \in \{35, 70\}) \times (N_r \in \{151, 301\})$ **4개 조합 전부**에서 에지 온도, 에지 구배($dT_{\text{1D}}/dr$), 최종 승인 여유를 비교한다.

### 3.3 PyTorch `float64` 1D 축대칭 웨이퍼 FVM & Adjoint Newton
* $N_r = 151$ 비등간격 셀($r_{-1/2}=0$ 영 면적), 셀 중심–경계면 거리($\delta_j = r_{j+1/2} - r_j$, $\delta_{j+1} = r_{j+1} - r_{j+1/2}$) 가중 조화평균 $k_{\text{Si}}|_{j+1/2}$, 최외곽 반 셀 복사면 온도 $T_{\text{1D},N_r-1}$(FEM 외벽 대비 오차 검증).
* **행별 무차원 Newton 정지 기준:** $\|\mathbf{r}_w / q_{\text{ref}}\|_\infty < 10^{-11}$ 및 $\|\mathbf{r}_{\text{ESC}} / T_{\text{ref}}\|_\infty < 10^{-11}$ 동시 충족. 허용 영역($T_{\text{1D}} > 200\text{ K}$) 내 백트래킹 선탐색을 적용하며 미수렴 상태에는 Implicit Diff를 적용하지 않고 실패 플래그를 반환한다.
* **Adjoint Implicit Diff:** 수렴점 $\mathbf{x}^*$에서 $(\nabla_{\mathbf{x}}\mathbf{F})^T \boldsymbol{\lambda} = \mathbf{v}$를 `torch.linalg.solve` 1회로 풀어 VJP 계산. **① 구현 검증(Adjoint vs 동일 FVM 중앙유한차분, 0 부근 절대오차 병용)**과 **② 축약 모델 감도 검증(FVM Adjoint vs 2D FEniCSx 유한차분)**을 분리 보고한다.

---

## 4. 2단계 역문제: 활성 인터락 최우선 차단, 전 규격 마진 전파 및 $A_{\text{pre}}/V$ 분리

### 4.1 활성 장비 정지 인터락($H_{\text{block}}$) 최우선 게이트 및 Stage 1 진단
* **활성 정지 인터락 신호 ($H_{\text{block}} \in \{0, 1\}$):** 활성 OEM He Flow High 알람, 배면 압력 추종 불능, 클램핑 이상, 또는 필수 히터/압력 Readback 미확인 시 $H_{\text{block}} = 1$. **$H_{\text{block}} = 1$이면 탐색점 $\mathbf{u}_{\text{probe}}$ 인가와 최종 사전 승인($A_{\text{pre}}$)을 모두 즉시 차단**하고 `Hold / PM Review`로 분기한다(과거 이력성·예방적 PM 권고는 비차단 로그로 별도 기록).
* **관측 벡터 및 유량 Model Mismatch:** $\mathbf{y} = (T_1, \dots, T_9, \dot{m}_{\text{He}})$. Ground-Truth 유량 생성기는 비선형 점성 누설항 $\propto (1 - \beta_{\text{leak}}^2)P_{\text{out}}^2\cdot(1 + \delta_{\text{cal}})$($\pm 20\%$ 드리프트)을, 추정기는 선형 근사를 사용한다.
* **1차 적합 (기준점 $\mathbf{u}_0 \in \mathcal{U}_{\text{HW}}$):** $S_0 = \min_{\mathbf{d}\in\mathcal{D}}(\mathbf{y}_0 - f(\mathbf{u}_0, \mathbf{d}))^T \mathbf{R}_0^{-1}(\mathbf{y}_0 - f(\mathbf{u}_0, \mathbf{d}))$.
* **2차 방어 — 차분 관측 기반 Pre-Refit 게이트 ($H_{\text{block}} = 0$일 때만 실행):**
  * $\hat{\mathbf{d}}$를 재적합하지 않고 고정한 채, 장비 허용범위 및 냉각 여유가 확인된 탐색점 $\mathbf{u}_{\text{probe}} \in \mathcal{U}_{\text{HW}}$에서의 변화량 $\Delta\mathbf{y} = \mathbf{y}_{\text{probe}} - \mathbf{y}_0$와 예측 변화량 $\Delta f(\hat{\mathbf{d}}) = f(\mathbf{u}_{\text{probe}}, \hat{\mathbf{d}}) - f(\mathbf{u}_0, \hat{\mathbf{d}})$의 오차를 평가한다:
    $$S_{\text{probe}} = \big(\Delta\mathbf{y} - \Delta f(\hat{\mathbf{d}})\big)^T \mathbf{R}_{\Delta}^{-1} \big(\Delta\mathbf{y} - \Delta f(\hat{\mathbf{d}})\big)$$
    여기서 $\mathbf{R}_{\Delta}$는 $\hat{\mathbf{d}}$ 추정 오차와 공통 드리프트를 보존한 쌍(Pair) 단위 캘리브레이션 자료로 보정한다.
  * $S_0$, $S_{\text{probe}}$, 추정 불확실성($\Sigma_{\mathbf{d}}$) 중 하나라도 문턱을 초과하면 진단 보류(`Hold`).

### 4.2 Stage 2: 전 규격 마진 전파, 시나리오·피복 상계 $\overline{\rho}(\mathbf{u}^*)$ 및 $A_{\text{pre}}/V$ 분리
* **평가 영역이 엄격히 분리된 제약 집합 $g_i(\mathbf{u}, \xi) \le 0$:**
  1. **장비 허용범위 ($\mathbf{u} \in \mathcal{U}_{\text{HW}}$):** $Q_i \in [0, 400]\text{ W}$, $P_{\text{in}}, P_{\text{out}} \in [3.0, 20.0]\text{ Torr}$.
     * **누설 인터락:** 측정 He 유량이 정상 상한을 초과하거나 $\hat{\beta}_{\text{leak}} < 0.90$인 경우 $P_{\text{out}} \in [3.0,\ P_{\text{out},0}]\text{ Torr}$로 상한만 동결(하한 $3.0\text{ Torr}$ 유지). 누설 여부 미확정 시 `Hold`.
  2. **제품 웨이퍼 품질 규격 (유효 웨이퍼 면적 $r \le R_{\text{eff}} = 147\text{ mm}$, 3 mm 에지 제외폭 — 1D–상면 보수 마진 $\delta_T, \delta_\nabla$를 4개 규격 전부에 전파):**
     $$\begin{aligned}
     |\bar{T}_{\text{1D},A} - T_{\text{target}}| + \delta_T &\le 0.5^\circ\text{C}, \\
     (\max T_{\text{1D}} - \min T_{\text{1D}}) + 2\delta_T &\le 1.5^\circ\text{C}, \\
     3\sigma_A(T_{\text{1D}}) + 3\delta_T &\le 1.0^\circ\text{C}, \\
     \|\partial_r T_{\text{1D}}\|_\infty + \delta_\nabla &\le 0.3\ ^\circ\text{C/mm}\ \ (\text{벡터/평활 제약}).
     \end{aligned}$$
  3. **세라믹 보호 제약 (전체 ESC 보호 영역 $r \in [0, 155]\text{ mm}$):** $|d\hat{T}_{\text{ESC}}/dr|_j \le 1.5\ ^\circ\text{C/mm}$ (벡터/평활 제약), $|q''_i - q''_{i\pm 1}| \le 2.0\text{ W/cm}^2$.
  4. **칠러 제열 한계 (전체 베이스 하부 $r \in [0, 160]\text{ mm}$ 독립 적분 응답):** $Q_{\text{cool}}(\mathbf{u}, \xi) \le 1{,}800\text{ W}$.
* **최악 제약 위반도 상계 $\overline{\rho}(\mathbf{u}^*)$ 및 2단계 최적화:**
  * 불확실성 집합 $\mathcal{U}(\hat{\mathbf{d}})$를 피복 반경 $h$의 유한 시나리오 격자 $\{\xi_j\}_{j=1}^{N_s}$로 덮고, 시나리오 상의 Adjoint 기울기 노름으로부터 추정한 보수적 국소 Lipschitz 상수 $\hat{L}_i$를 더해 최악 위반도 상계를 정의한다:
    $$\overline{\rho}(\mathbf{u}) = \max_i \frac{\max_{1 \le j \le N_s} g_i(\mathbf{u}, \xi_j) + \hat{L}_i h}{s_i}$$
  * Epigraph 슬랙 변수를 둔 다중 시작점 SLSQP로 $\overline{\rho}(\mathbf{u})$를 먼저 최소화하고, $\overline{\rho}(\mathbf{u}) \le 0$을 달성한 실현 가능 영역 내에서 평활 목적함수($w_1\text{Var}_A + w_2(\bar{T}_A - T_{\text{target}})^2 + w_3\tau\,\mathrm{LogSumExp} + w_4\|\mathbf{u}-\mathbf{u}_0\|_W^2$)를 최소화해 최종 후보 $\mathbf{u}^*$를 확정한다.
* **사전 승인($A_{\text{pre}}$) 동결 vs 사후 독립 FEM 채점($V$)의 엄격 분리:**
  $$A_{\text{pre}} = \mathbf{1}\Big\{H_{\text{block}} = 0,\ \text{Stage 1 통과},\ \text{필수 Readback·압력 추종 확인},\ \overline{\rho}(\mathbf{u}^*) \le 0\Big\}$$
  $$V = \mathbf{1}\Big\{\text{숨겨둔 실제 상태 }\mathbf{d}_{\text{true}}\text{의 2D FEM 상면/전체 도메인에서 }\mathbf{u}^*\text{가 원래 제품 규격(마진 미포함)·설비 제약 위반}\Big\}$$
  * 장비 가용 정보만으로 **$A_{\text{pre}}$와 $\mathbf{u}^*$를 먼저 기록·동결한 뒤**, 숨겨둔 실제 상태(3개 파라미터 + 누설 위치 + 축대칭 이중 반경 피크 `bimodal_rf` + 휨·leak-by 유량 결합)의 2D FEniCSx FEM으로 $V$를 사후 채점한다.
  * **3단계 운영 판정:**
    1. **사전 승인 (`Pre-Approved`, $A_{\text{pre}}=1$):** $H_{\text{block}}=0$ 및 모든 진단·강건 상계 통과.
    2. **복구 미확인 / 보류 (`Unverified / Hold`, $A_{\text{pre}}=0$):** $H_{\text{block}}=1$, 게이트 차단, Readback 미확인, 또는 다중 시작점 SLSQP에서 $\overline{\rho} > 0$ → 수동 점검·재진단.
    3. **PM 검토 권고 (`PM Review`):** Stage 1 적합값의 재확인이 아닌 **독립 하드웨어 알람/점검 증거**(활성 $H_{\text{block}}$ 유발 고장 또는 별도 리크 체크 등)가 하드웨어 이상을 지지할 때 별도 기록·권고.
  * **독립 위험 통계 보고:** 전용 캘리브레이션 세트에서 문턱을 고정한 뒤, 독립 드로우(드리프트 독립 샘플링, 정상·고장·OOD 혼합비 명시) 테스트 세트에서 승인율과 조건부 오승인율 $\hat{P}(V=1 \mid A_{\text{pre}}=1)$을 **사전 승인 건수 $n_A = \sum A_{\text{pre}}$를 분모로 한 단측 95% Clopper-Pearson 신뢰상한**과 함께 보고한다($n_A = 0$이면 추정 불가 명시).

---

## 5. 2주 MVP 및 Phase 2 확장 로드맵

### 2주 MVP ($\text{Al}_2\text{O}_3$ 단일 소재 — 물리 검증 & 사전승인/사후채점 시험표 완성)
* **Week 1 (Step 1 — 물리·4조합 수렴·독립 열수지·Adjoint 검증):**
  * Day 1: `esc_fem_solver.py` 단건·배치 실행 시간 실측 → $M=70$ 응답 구축, 감도 유한차분, 사례별 기준·탐색·최종 FEM 호출을 포함한 **총 계산 예산 확정**.
  * $(M \in \{35, 70\}) \times (N_r \in \{151, 301\})$ 4조합 수렴, 웨이퍼 및 독립 Robin 제열 열수지 $< 10^{-10}\text{ W}$, 유한 검증 운전영역에서의 $\delta_T, \delta_\nabla$ 실측, Adjoint 구현/감도 분리 Gradcheck, 추론 시간 중앙값·p95 보고.
* **Week 2 (Step 2 — 사전승인 vs 사후 FEM 독립 시험표 및 핵심 시연):**
  * 캘리브레이션 세트 문턱 고정 → 독립 테스트 세트에서 **`[사례별 H_block, A_pre, 승인 제어축, 적용 장비범위, 사후 2D FEM 원규격 판정 V, 실제 위반 위치, Hold/PM 근거]`**를 나란히 기록한 **독립 시험표** 완성.
  * 숨겨둔 외란 2D FEM에서 ① 복구 성공 사례와 ② 단일점 적합도는 좋았으나 Pre-Refit 게이트가 오진을 차단한 사례를 End-to-End 추적, 고정 조건 명시 하에 $(\alpha_{\text{peak}}, \beta_{\text{leak}})$ 맵 작성.

### Phase 2 확장 (MVP 완료 후 — 2단 성적표 공정 Ablation & $\text{AlN}$ 비교)
* 동일 LHS(100/300/500개, 훈련 전용 POD, 라벨 명시) 기반 **2단 성적표** 비교:
  1. **4종 공통 성적표** (`Data-only MLP`, `Soft-PINN`, `Linear ESC POD+Ridge + Diff-1D-FVM`, `Neural ESC Mode B + Diff-1D-FVM`): 동일 추론 입력 기준 $T_w$ 면적 MAE, 에지 오차, $\partial T_w/\partial(\mathbf{u}, \mathbf{d})$, 독립 FEM 규격 충족률, 분리 외삽($\beta_{\text{leak}} < 0.15$ 및 저-$P_{\text{out}}$).
  2. **ESC 연산자 쌍 비교 성적표** (`Linear POD+Ridge` vs `Neural Mode B`, 동일 $(T_{\text{ESC}}, q''_{\text{gap}})$ 감독): 비평형 계면 하중 교란 방향미분 $\partial T_{\text{ESC}}/\partial q''_{\text{gap}}$ 및 Newton 실패율.
* **$\text{AlN}$** 소재 교체(Mode A 응답행렬 재추출 및 Mode B 재학습) 후 $\text{Al}_2\text{O}_3$ vs $\text{AlN}$ 비교.
