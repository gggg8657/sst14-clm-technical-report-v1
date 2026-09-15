# R1: 기존 시스템 비교 분석

**작성일**: 2026-06-17  
**작성자**: researcher  
**목적**: SSTR2 선택성 방사성의약품 펩타이드 발굴 파이프라인의 객관적 포지셔닝 — 기존 도구 대비 차별점 및 한계 정직하게 기술

---

## 검색 전략

| 검색어 | 목적 |
|--------|------|
| RFdiffusion ProteinMPNN peptide binder design 2024 2025 | 펩타이드/단백질 바인더 설계 파이프라인 현황 |
| AlphaProteo DeepMind protein binder design 2024 | AlphaProteo 방법론 및 성능 |
| BindCraft protein binder design pipeline 2024 2025 | BindCraft 접근법 확인 |
| FlexPepDock Rosetta peptide docking affinity prediction | 도킹 도구 비교 |
| AlphaFold3 Boltz DiffDock peptide docking binding affinity | 구조 예측 기반 도킹 |
| PEPlife server peptide half-life prediction | 반감기 예측 도구 |
| pepADMET ADMET prediction peptide drug 2024 | 펩타이드 특화 ADMET |
| ADMETlab 3.0 admet-ai 2024 | 소분자 ADMET 플랫폼 |
| Coscientist ChemCrow autonomous AI agent 2023 2024 | Agentic AI 과학 사례 |
| autonomous AI science agent drug discovery loop 2024 2025 | 자율 약물 발굴 에이전트 |

---

## 1. 펩타이드/단백질 바인더 설계 파이프라인

| 도구 | 접근 방식 | 강점 | 우리 시스템과의 차이 | 출처 URL |
|------|-----------|------|---------------------|----------|
| **RFdiffusion + ProteinMPNN** | 확산 모델로 backbone 생성 → ProteinMPNN 시퀀스 설계 → AF2 검증 | de novo 바인더 설계 성공률 높음; pAE_interaction < 10 필터링; 15,000개 서열 라이브러리 생성 가능 | **타겟 비교**: 구조 미지 단백질에도 적용 가능하나 소분자 기반 방사성의약품(DOTA 라벨링) 최적화는 범위 밖. 선택성(SSTR2 vs 다른 SSTR 서브타입) 다목적 스코어링 없음. | [GitHub RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) |
| **AlphaProteo** (DeepMind, 2024) | AF 계열 ML 모델; 기존 단백질 결합 포켓에 de novo 바인더 설계; 단일 라운드 스크리닝으로 즉시 사용 가능한 바인더 제공 | 기존 대비 바인딩 친화도 3~300배 향상; VEGF-A, SARS-CoV-2 spike 등 어려운 타겟 성공; 실험 성공률 10배 (BHRF1 기준) | **접근 불가**: 제한적 Trusted Tester Program(학계/산업계 한정). 상업적 응용은 Isomorphic Labs 전담. 오픈소스 아님 → 우리 파이프라인 통합 불가. TNFα 등 일부 타겟 실패 | [arxiv 2409.08022](https://arxiv.org/abs/2409.08022), [HLTH 뉴스](https://hlth.com/insights/news/google-deepmind-launches-alphaproteo-to-advance-protein-binder-design-2024-09-11) |
| **BindCraft** (2024→Nature 2025) | AF Multimer로 backbone+시퀀스 공동 최적화; MPNN_sol 최적화; AF2 monomer 필터; 원샷(one-shot) 설계 | 실험 성공률 10~100%; 나노몰 친화도 달성; 결합 부위 미지에도 적용; MIT 라이선스 오픈소스 | 단백질 바인더 설계 특화(미니단백질 크기). SST-14 같은 14aa 고리형 펩타이드 → 선택성 최적화를 위한 다목적 스코어링 없음; DOTA 킬레이션 적합성 미고려 | [Nature 2025](https://www.nature.com/articles/s41586-025-09429-6), [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.09.30.615802v1) |

**핵심 추출**:
- RFdiffusion+ProteinMPNN은 backbone 생성부터 시퀀스 설계까지 통합하지만, 수용체 서브타입 선택성(SSTR2 vs SSTR1/3/4/5)의 동시 다목적 최적화는 설계 목표에 없음.
- AlphaProteo는 성능이 가장 높지만 클로즈드 시스템; 방사성의약품 특화 기능 없음.
- BindCraft는 오픈소스이며 성능 우수하나, 소분자 킬레이터 적합성·반감기 최적화는 스코프 밖.

---

## 2. 도킹 / 결합 친화도 예측

| 도구 | 접근 방식 | 강점 | 우리 시스템과의 차이 | 출처 URL |
|------|-----------|------|---------------------|----------|
| **FlexPepDock** (Rosetta) | 물리 기반 Rosetta 에너지 함수; ab-initio 동시 folding+docking+refinement | r=0.59 친화도-에너지 상관; 25개 벤치마크 중 20개 ≤1.5Å 재현; 비표준 아미노산(phospho, D-aa) 지원 | 우리 시스템(PyRosetta)은 FlexPepDock 기반이지만, **선택성 ΔΔG = ΔG(SSTR2) − ΔG(off-target)** 계산 및 home-advantage 보정을 추가 구현. FlexPepDock 단독으로는 선택성 비교·다목적 스코어링 없음 | [PubMed 21572516](https://pubmed.ncbi.nlm.nih.gov/21572516/), [PubMed 36512534](https://pubmed.ncbi.nlm.nih.gov/36512534/) |
| **AlphaFold3 / Boltz** | 딥러닝 구조 예측 기반 도킹; Boltz-2(2025.06)에서 binding affinity 예측 추가 | 단백질-펩타이드 binding geometry 고정밀 예측; Boltz-2는 친화도까지 예측 시도 | 훈련 bias 우려: 기학습 구조에 편향 (Guan 2025 Protein Science). 새 단백질/결합 부위 일반화 한계. 우리 시스템은 **실측 PyRosetta ΔG** 사용 → surrogate 모델 아님 | [Boltzina arxiv](https://arxiv.org/pdf/2508.17555), [Protein Science 2025](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.70331) |
| **DiffDock** | 확산 모델 기반 단백질-리간드 도킹; 소분자 위주 | 비정형 결합 포즈 탐색 우수; 유연한 사이드체인 처리 | 주로 소분자 리간드 대상. 펩타이드-GPCR 도킹은 CABS-dock+FlexPepDock 조합이 더 적합(문헌 근거). Selectivity margin 계산 미지원 | [ChemRxiv 2025](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2025-b8k76) |

**핵심 추출**:
- 우리 시스템의 핵심 차별점: PyRosetta 실측 ΔG를 SSTR2와 off-target(SSTR1/3/4/5) 양쪽에 모두 계산하여 **Δmargin(선택성 home-advantage)** 스코어 산출. 이는 단순 도킹 점수 최적화를 넘어 "SSTR2에 선택적인가"를 직접 최적화.
- AlphaFold3/Boltz 계열은 빠르고 geometry 예측 우수하지만 훈련 편향 위험이 보고됨; 우리 파이프라인은 물리 기반 Rosetta를 앵커로 사용.

---

## 3. 혈중 반감기 예측 도구

| 도구 | 접근 방식 | 강점 | 우리 시스템과의 차이 | 출처 URL |
|------|-----------|------|---------------------|----------|
| **PEPlife** (2016) / **PEPlife2** (2025 preprint) | 실험 측정 반감기 큐레이션 데이터베이스(in vivo 948건 + in vitro 1265건); HPLC/RIA/ELISA 기반 | 최초 포괄적 펩타이드 반감기 저장소; ML 예측 도구 개발 기반 제공; PEPlife2로 업데이트 | 우리 시스템의 반감기 스코어는 **surrogate(대리) 지표**(이화학적 특성 기반 휴리스틱) → PEPlife 실측값과 직접 보정되지 않음. **한계 인정 필수**: 실험 검증 전까지는 근사값 | [Sci Rep 2016 PMC5098197](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5098197/), [PEPlife2 bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.13.653654v1) |
| **In silico 반감기 예측** (Miklius 2018 등) | ML 모델; 시퀀스·변형·이화학적 특성 기반 혈중 안정성 예측 | 변형된 펩타이드(D-aa, 메틸화) 반감기 예측 시도 | 우리 파이프라인에 통합되지 않음. 통합 시 surrogate 정확도 개선 가능 (§검증 필요) | [PLOS One 0196829](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196829) |

**핵심 추출**:
- 펩타이드 반감기 예측은 여전히 실험 데이터 부족·방법론 이질성 문제가 있음. PEPlife2가 최신 큐레이션이지만 예측 도구 자체는 아직 프리프린트 단계.
- 우리 시스템의 반감기 추정은 **surrogate임을 명시적으로 인정해야 함**; 실험 검증 전 의사결정에 과도 의존 금지.

---

## 4. ADMET 예측 플랫폼

| 도구 | 접근 방식 | 강점 | 우리 시스템과의 차이 | 출처 URL |
|------|-----------|------|---------------------|----------|
| **pepADMET** (2025) | 최초 펩타이드 특화 AI-driven ADMET 플랫폼; MLR-GAT 프레임워크; 선형/고리형/변형/천연 펩타이드 동시 지원; 종·기관·세포주별 생물학적 변이 반영 | 펩타이드 독성 다범주 계층적 예측 (MLR-GAT); 고리형 펩타이드 직접 지원; 온라인 무료 접근 | 우리 시스템에서 ADMET 예측은 surrogate 지표 사용. pepADMET API 통합 시 실질적 개선 가능. **단**: pepADMET 출판(JCIM 2025)이 최신이어서 벤치마크 독립 검증 미완 | [JCIM 2025 doi](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02518), [Semantic Scholar](https://www.semanticscholar.org/paper/pepADMET:-A-Novel-Computational-Platform-For-ADMET-Tan-Liu/fa070ba4ffa0b553cf38e7b2d7079bfe030882af) |
| **ADMETlab 3.0** (2024) | DMPNN + molecular descriptors 멀티태스크; 119 endpoints; 40만 건 데이터; API 지원 | 소분자 커버리지 가장 넓음; ROC 및 회귀 성능 우수; 구글 스칼라 인용 955회(v2.0) | **소분자 특화**: 펩타이드 직접 입력 미지원. SSTR2 타겟 방사성의약품 펩타이드에 적용 시 정확도 저하 가능 | [Nucleic Acids Research 2024](https://academic.oup.com/nar/article/52/W1/W422/7640525), [PMC11223840](https://pmc.ncbi.nlm.nih.gov/articles/PMC11223840/) |
| **ADMET-AI** (2024) | Chemprop-RDKit GNN; TDC 41개 데이터셋 학습; TDC 리더보드 평균 1위; 100만 분자/3.1시간 | TDC 벤치마크 최고 순위; 오픈소스 Python 패키지; 대규모 라이브러리 고속 처리 | 소분자 전용; 펩타이드 직접 지원 없음. ADMET-AI 적용 시 펩타이드를 SMILES로 변환 필요하며 정확도 불확실 | [Bioinformatics 2024](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030), [PMC11226862](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11226862/) |

**핵심 추출**:
- 우리 시스템의 ADMET 추정은 **surrogate 지표**(이화학적 특성 기반)이며, 이는 pepADMET 등 전용 도구 대비 정확도 열세를 인정해야 함.
- pepADMET(2025)은 고리형 펩타이드를 직접 지원하는 유일한 AI ADMET 도구이며, 향후 파이프라인 통합 후보로 가장 적합.
- ADMETlab 3.0 및 ADMET-AI는 소분자 특화이므로 SST-14 변이체에 직접 적용 시 한계 있음.

---

## 5. Agentic AI for Science

| 시스템 | 접근 방식 | 강점 | 우리 시스템과의 차이 | 출처 URL |
|--------|-----------|------|---------------------|----------|
| **Coscientist** (GPT-4 기반, 2023) | GPT-4 + 인터넷/문서 검색 + 코드 실행 + 실험 자동화; 6개 과제 검증 (팔라듐 크로스커플링 최적화 등) | 완전 자율 실험 설계·실행; paracetamol·aspirin 합성 성공; 실물 로보틱스 연동 | 화학합성 실험 자동화 특화. 우리 시스템은 **인실리코(in silico)** 발굴 루프 특화. 실물 로보틱스 없음. SSTR2 선택성 스코어링 도메인 없음 | [Nature 2023](https://www.nature.com/articles/s41586-023-06792-0) |
| **ChemCrow** (GPT-4 + 18 도구, 2023) | 18개 전문가 설계 도구 통합 LLM 에이전트; 유기합성·약물발굴·재료설계 | 인간 평가 9.24/10 (GPT-4 단독 4.79/10); 신규 크로모포어 발굴; 3개 유기촉매 합성 | 유기화학 합성 특화. 방사성의약품 GPCR 타겟 선택성 최적화 루프 미포함 | [PMC11116106](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11116106/) |
| **DrugAgent** (LLM 다중 에이전트, 2024) | LLM 다중 에이전트 협업; Drug-Target Interaction(DTI) 예측 자동화; ReAct 대비 ROC-AUC +4.92% | 약물-타겟 상호작용 코드 자동 생성·실행; 재현 가능한 파이프라인 | DTI 예측 자동화 특화. 선택성 다목적 스코어링·무한 루프 발굴·방사성의약품 특화 없음 | [arxiv 2411.15692](https://arxiv.org/abs/2411.15692) |
| **우리 시스템** (PRST_N_FM) | PyRosetta 실측 도킹(SSTR2 + off-target) + LLM 에이전트 오케스트레이션 + 무한 발굴 루프 + 다목적 스코어링(선택성 Δmargin·독성·결합) | **SSTR2-특화** 선택성 home-advantage 계산; 무한 자율 발굴 루프(continuous.py); 실측 물리 기반 ΔG; 방사성의약품 DOTA 적합성 검토 포함 | — | 내부 시스템 |

**핵심 추출**:
- Coscientist·ChemCrow는 합성화학 실험 자동화에 특화; 우리 시스템은 인실리코 발굴 루프 특화.
- 기존 Agentic AI 시스템들은 **단일 도메인 도구 집합** 중심; 우리 시스템은 **물리 기반 도킹(PyRosetta) + LLM 에이전트 + 무한 루프**의 수직 통합.
- DrugAgent(2024)와 가장 유사한 구조이나, DrugAgent는 DTI 분류 자동화 반면 우리는 선택성 ΔG 최적화에 집중.

---

## 종합 포지셔닝

### 우리 시스템의 실질적 차별점

| 차별점 | 근거 | 신뢰 등급 |
|--------|------|-----------|
| **실측 ΔG 기반 선택성 스코어** — PyRosetta로 SSTR2·off-target 양쪽 ΔG 계산 후 Δmargin(home-advantage) 산출 | 기존 도구들(AlphaProteo, BindCraft 등)은 단일 타겟 최적화; 서브타입 선택성 다목적 최적화 없음 | HIGH |
| **방사성의약품 특화 도메인** — DOTA 킬레이션 적합성·방사성동위원소 theranostics 맥락 | 일반 단백질 바인더 설계 파이프라인에 없는 도메인 | HIGH |
| **무한 자율 발굴 루프** — continuous.py 기반 멈추지 않는 후보 생성·평가·누적 | Coscientist/ChemCrow는 단회성 실험 자동화; 지속적 탐색 루프 구조 다름 | HIGH |
| **PyRosetta + LLM 하이브리드** — 물리 기반 앵커 + 에이전트 오케스트레이션 | Boltz/AF3 surrogate 방식 대비 물리 기반 신뢰성 높음 | MED (실험 검증 전) |

### 우리 시스템의 인정된 한계

| 한계 | 비교 대상 우세 도구 | 신뢰 등급 |
|------|---------------------|-----------|
| **반감기 예측은 surrogate 지표** — 실험 측정값(PEPlife 기준)과 보정 없음 | PEPlife 실측 데이터베이스 | HIGH (한계 확실) |
| **ADMET는 이화학적 특성 기반 근사** — pepADMET, ADMET-AI 수준의 ML 기반 예측 미통합 | pepADMET, ADMETlab 3.0 | HIGH (한계 확실) |
| **de novo 바인더 설계 기능 없음** — 변이(mutation) 탐색 방식; RFdiffusion 같은 backbone 설계 미포함 | RFdiffusion, BindCraft | MED |
| **실험 검증 미완** — 인실리코 예측값만 존재; 결합 친화도·선택성 실험 데이터 없음 | AlphaProteo(실험 성공률 보고) | HIGH (한계 확실) |

---

## §검증 필요

1. **PEPlife2 예측 도구 포함 여부**: bioRxiv 2025.05.13 프리프린트 접근 실패(403). PEPlife2가 단순 데이터베이스 업데이트인지, ML 예측 도구 포함인지 확인 필요.
2. **pepADMET 고리형 펩타이드 성능 벤치마크**: JCIM 2025 논문 paywall. 독립 벤치마크 데이터 미확보. SST-14 같은 이황화결합 고리형 펩타이드 적용 정확도 미확인.
3. **AlphaFold3 bias 정량적 영향**: Guan 2025(Protein Science)에서 훈련 편향 보고했으나, SSTR2-SST-14 계열에서의 실제 오차 크기 미추정.
4. **DrugAgent vs 우리 시스템 직접 비교**: DTI ROC-AUC 지표 vs 우리의 Δmargin 지표가 다른 평가 공간이라 직접 비교 불가. 공통 벤치마크 미존재.
5. **CycleRFdiffusion (2025 preprint)**: 고리형 펩타이드 특화 확산 모델로 TREM2 타겟 적용 사례 보고. biorxiv 2025.09 접근 불가(미래 날짜 — 출처 메타데이터 오류 가능성). SSTR2 적용 가능성 재검토 필요.

---

*인용 형식: (저자 연도 저널 또는 DOI/URL), 모든 URL은 조회일 2026-06-17 기준.*
