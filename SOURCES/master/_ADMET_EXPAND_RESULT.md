# ADMET 용혈 벤치마크 확장 결과 기록

생성일: 2026-06-18

---

## 데이터셋 확장 개요

| 항목 | 이전 | 이후 |
|------|------|------|
| 총 종수 | 26종 | **52종** |
| L-aa | 22종 | **33종** |
| D-aa | 4종 | **19종** |
| 평가 가능 수 (cyclic 제외) | 24종 | **50종** |

- 신규 추가 26종: D-aa 대폭 보강 (4→19). D-Piscidin 변이체 시리즈(I9K, G8P, V12K, G13K), D-Dermaseptin 변이체 시리즈(L7K, A14K, L7K+A14K), D-RR4, DGL13K, TPW-1, TPW-3, Protopolybia-MPIII, Polybia-MPIII, Pm-R3, PDD-B, Agelaia-MPII, Mastoparan-J, PMM2, Mastoparan-Lew, Protonectarina-MP, Pm-R1, Mastoparan-T3, Mastoparan-II, D1~D5 시리즈. 전 항목 문헌 출처 보유.
- 상세 목록: `_analysis/BENCH_admet_v2.md` 참조.

---

## AUC Before → After 비교

| 그룹 | 이전 AUC (n) | 이후 AUC (n) | 변화 |
|------|:---:|:---:|------|
| 전체 | 0.193 (L20/D4, 24평가) | **0.32** (L32/D18, 50평가) | 완화(역변별 지속) |
| L-aa | 0.167 (n=20) | **0.146** (n=32) | 역변별 유지·견고화 |
| D-aa | 0.25 (n=4, 과소) | **0.677** (n=18) | 양의 변별 확인 |

원본 결과 파일: `_analysis/BENCH_admet_v2_results.json`

---

## 핵심 해석 (정직)

**(a) D-aa 표본 4→18 확보로 변별력 확인**: D-aa 그룹 AUC 0.677 (>0.5 = 양의 변별). 이전 D-aa AUC 0.25는 n=4 소표본 노이즈였으며, 충분한 표본에서는 pepADMET이 D-aa 용혈 분류에 변별력이 있음을 확인.

**(b) L-aa는 AUC 0.146으로 역변별 — 견고히 입증된 진짜 한계**: n=32로 늘어도 AUC가 0.167에서 0.146으로 오히려 낮아져, L-aa 용혈/비용혈 역전 양상이 노이즈가 아닌 실질적 한계임을 추가 확인. "막활성 AMP 탐지기"에 가깝고 용혈 특이성이 약하다는 기존 해석 유지.

**(c) 전체 0.32도 역변별이나 완화**: L-aa 역변별이 지배적이나, D-aa 변별력 편입으로 전체 AUC가 0.193 → 0.32로 소폭 상승. 그러나 0.5 미만이므로 절대 독성 예측기로서의 신뢰 불가 결론은 유지.

---

STATUS: DONE
