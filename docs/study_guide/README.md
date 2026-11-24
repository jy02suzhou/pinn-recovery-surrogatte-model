# 3D ESC 미분 가능 물리 & 이상치 복구 — 학부 4학년 학습 가이드

> **대상 독자:** 기계공학과 학부 4학년 (열전달 · 선형대수 · 수치해석 · 자동제어 기초)
> **예상 완독 시간:** 약 2~3시간 (챕터당 15~25분, 총 7개 챕터 + 치트시트/면접 5선)
> **읽는 순서:** Ch 1부터 순서대로 읽으며 각 장의 **직관 비유 → 핵심 수식 → 실제 코드 & 그래프 → 자가 점검 퀴즈**를 체크하세요.

---

## 학습 로드맵 및 완독 체크리스트

| 순서 | 챕터 | 핵심 주제 | 예상 시간 | 연결 코드 / 그래프 | 완료 |
| :---: | :--- | :--- | :---: | :--- | :---: |
| **1** | [Ch 1. 큰 그림 먼저 잡기](ch01.md) | 3D 10존 ESC 열평형, 수동 보정의 한계, Phase 1 vs 2-A FAQ | 20분 | `build_esc_3d_mesh.py` · `day1_3d_wafer_demo.png` | `- [ ]` |
| **2** | [Ch 2. 반도체 ESC와 희박 헬륨](ch02.md) | Kennard–Yovanovich($d+M(P)$), 누설 시 승압 금지(척킹·아킹) | 20분 | `materials_and_he.py` · `esc_3d_fem_solver.py` | `- [ ]` |
| **3** | [Ch 3. 속도 27배: ROM + 2D FVM](ch03.md) | 선형 ESC 중첩(Mode A), $\text{Bi}_z \ll 1$, Schur 소거(대입법) | 25분 | `esc_3d_unit_load.py` · `coupled_3d_wafer_fvm.py` · `week2_3d_recovery_showcase.png` | `- [ ]` |
| **4** | [Ch 4. 기울기 514배: Adjoint 미분](ch04.md) | 유한차분(24회) vs 수반법(1회), `float64` & Gradcheck(`1.02e-9`) | 20분 | `coupled_3d_wafer_fvm.py` · `step1_3d_hybrid_verify.py` | `- [ ]` |
| **5** | [Ch 5. 역문제와 Pre-Refit 오진 차단](ch05.md) | 정규화 오차($S_0, S_{\text{probe}}$), `test_15` 웨이퍼 휨 단독 차단 | 20분 | `stage1_3d_diagnosis.py` | `- [ ]` |
| **6** | [Ch 6. 안전 역설계 & 통계 보증](ch06.md) | SLSQP, Operable Envelope($\beta \ge 0.68$), $A_{\text{pre}}/V$, Clopper–Pearson(`4.31%`) | 25분 | `stage2_3d_recovery.py` · `phase15_verification_suite.py` · `phase15_operable_envelope.png` | `- [ ]` |
| **7** | [Ch 7. PINN은 어디로 가는가?](ch07.md) | `MLP` vs `Soft-PINN` vs `Hybrid Mode A/B` & Phase 2-A/2-B 로드맵 | 15분 | `coupled_3d_wafer_fvm.py` | `- [ ]` |
| **부록** | [Appendix. 치트시트 & 면접 5선](appendix.md) | 핵심 용어 2줄 사전 · SSOT 수치 암기표 · 면접 질문 5선 | 15분 | 전체 SSOT ([notation_ssot.md](_build/notation_ssot.md)) | `- [ ]` |

---

## 파이프라인 검증 산출물 (SSOT & Audit)
* 전역 기호·수치 사전 (SSOT): [_build/notation_ssot.md](_build/notation_ssot.md)
* 챕터별 코드·그래프 매핑표: [_build/manifest.yaml](_build/manifest.yaml)
* Stage 4 품질 감사 리포트: [_build/audit_report.md](_build/audit_report.md)
* 상세 공학 결과 보고서: [../DETAILED_RESULTS_REPORT.md](../DETAILED_RESULTS_REPORT.md)
