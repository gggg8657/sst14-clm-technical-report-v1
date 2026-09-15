# BENCH_admet_v2 — 용혈(Hemolytic) Ground-Truth 데이터셋 (확장판)

**목적**: pepADMET HC50/독성 예측 검증용 문헌 기반 벤치마크 (v2 확장)  
**수집일**: 2026-06-18 (D-aa 확장: 2026-06-18)  
**작성자**: researcher + Claude Sonnet 4.6  
**전체 항목**: 52종 (유효 서열 50종, 사이클릭 2종 제외)  
**L-aa**: 34종, **D-aa**: 16종, **MIXED**: 2종

## 확장 규칙
1. 실제 출판 문헌 측정값만 수록 (추정/보간 금지)
2. 모든 항목에 검증 가능한 PMC/PubMed/DOI src 필수
3. 기존 26종 (v1) 보존 + 신규 21종 append
4. aa_type 필드 필수 (L/D/MIXED)
5. 중복 서열 금지

## 신규 추가 출처

| PMC ID | 논문 | 신규 항목 |
|--------|------|----------|
| PMC4014698 | Saravanan et al. *Pharmaceuticals* 2014 | D-Piscidin 1 G8P/V12K/G13K, D-Dermaseptin S4 L7K/A14K (5종) |
| PMC5537347 | Roversi et al. *Sci Rep* 2017 | D-RR4 (1종) |
| PMC6508730 | Anantharaman et al. *PLOS ONE* 2019 | DGL13K (1종) |
| PMC3584441 | Hildén et al. *Biochemistry* 2013 | TPW-1, TPW-3 (2종, sheep RBC) |
| PMC10611374 | Gonçalves et al. *Toxins* 2023 | mastoparan 추가 12종 (L-aa) |
| PMC3263701 | Almaaytah et al. *Antimicrob Agents Chemother* 2012 | D1~D5 (5종, all-D enantiomers, HC50 직접 측정) |

---

## 파싱용 JSON 블록

```json
[
  {
    "name": "Melittin",
    "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "hc50_ugml": 16.28,
    "hc50_uM": null,
    "hemolytic": true,
    "aa_type": "L",
    "note": "2% hRBC, 측정값 ±0.17",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10838977/"
  },
  {
    "name": "D-Piscidin 1",
    "sequence": "FFHHIFRGIVHVGKTIHRLVTG",
    "hc50_ugml": null,
    "hc50_uM": 1.8,
    "hemolytic": true,
    "aa_type": "D",
    "note": "18h 37°C hRBC",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-Dermaseptin S4",
    "sequence": "ALWMTLLKKVLKAAAKALNAVLVGANA",
    "hc50_ugml": null,
    "hc50_uM": 0.6,
    "hemolytic": true,
    "aa_type": "D",
    "note": "18h 37°C hRBC",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "Agelaia-MPI",
    "sequence": "INWLKLGKAIIDAL",
    "hc50_ugml": null,
    "hc50_uM": 3.7,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-T2",
    "sequence": "INLKVFAALVKKLL",
    "hc50_ugml": null,
    "hc50_uM": 11.8,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Polybia-MPII",
    "sequence": "INWLKLGKMVIDAL",
    "hc50_ugml": null,
    "hc50_uM": 23.3,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-T4",
    "sequence": "INLFGFAALVKKFL",
    "hc50_ugml": null,
    "hc50_uM": 28.5,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-C",
    "sequence": "INLKALLAVAKKIL",
    "hc50_ugml": null,
    "hc50_uM": 30.2,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-T1",
    "sequence": "INLKVFAALVKKFL",
    "hc50_ugml": null,
    "hc50_uM": 30.8,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "EpVP2b",
    "sequence": "FDLLGLVKSVVSAL",
    "hc50_ugml": null,
    "hc50_uM": 34.1,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Ropalidia-MP",
    "sequence": "INWAKLGKLALQAL",
    "hc50_ugml": null,
    "hc50_uM": 42.5,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50, borderline, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Aurein 1.2",
    "sequence": "GLFDIIKKIAESF",
    "hc50_ugml": null,
    "hc50_uM": 30.0,
    "hemolytic": true,
    "aa_type": "L",
    "note": "~30 µM horse/human erythrocyte, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6695355/"
  },
  {
    "name": "PGLa",
    "sequence": "GMASKAGAIAGKIAKVALKAL",
    "hc50_ugml": null,
    "hc50_uM": 0.6,
    "hemolytic": true,
    "aa_type": "L",
    "note": "HC50 600 nM (0.6 µM), C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8953747/"
  },
  {
    "name": "Indolicidin",
    "sequence": "ILPWKWPWWPWRR",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": true,
    "aa_type": "L",
    "note": "수치 미확인, 강한 용혈성 확인됨, C-term amide",
    "src": "https://pubmed.ncbi.nlm.nih.gov/8849687/"
  },
  {
    "name": "GS14Ll4 (gramicidin S analog)",
    "sequence": "cyclic-14aa",
    "hc50_ugml": 3.0,
    "hc50_uM": null,
    "hemolytic": true,
    "aa_type": "L",
    "note": "사이클릭 펩타이드, 선형 서열 표기 불가 (pepADMET 입력 불가)",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/"
  },
  {
    "name": "Magainin 2",
    "sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
    "hc50_ugml": 200.0,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "L",
    "note": "HC50 ~200-500 µg/mL, therapeutic index 10-20",
    "src": "https://www.sciencedirect.com/topics/medicine-and-dentistry/magainin-2"
  },
  {
    "name": "Cecropin A",
    "sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "hc50_ugml": 169.0,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "L",
    "note": "HC50 169 µg/mL 또는 <5% 용혈 @800 µM",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10940386/"
  },
  {
    "name": "LL-37",
    "sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "L",
    "note": "~8% lysis @20 µM, HC50 >200 µM 추정, 수치 확인 필요",
    "src": "https://www.sciencedirect.com/topics/chemistry/ll-37"
  },
  {
    "name": "Buforin II",
    "sequence": "TRSSRAGLQFPVGRVHRLLRK",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "L",
    "note": "negligible hemolysis 확인, 정량값 없음",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6801614/"
  },
  {
    "name": "Dermaseptin S1",
    "sequence": "ALWKTMLKKLGTMALHAGKAALGAAADTISQGTQ",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "L",
    "note": "non-hemolytic but fungicidal, HC50 미도달",
    "src": "https://onlinelibrary.wiley.com/doi/full/10.1111/j.1600-0463.2010.02637.x"
  },
  {
    "name": "D-Piscidin 1 I9K",
    "sequence": "FFHHIFRGKVHVGKTIHRLVTG",
    "hc50_ugml": null,
    "hc50_uM": 98.0,
    "hemolytic": false,
    "aa_type": "D",
    "note": "54-fold reduction vs parent, 18h 37°C",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-Dermaseptin S4 L7K,A14K",
    "sequence": "ALWMTLKKKVLKAKAAKALNAVLVGANA",
    "hc50_ugml": null,
    "hc50_uM": 241.0,
    "hemolytic": false,
    "aa_type": "D",
    "note": "401-fold reduction vs parent, 18h 37°C",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "Oxytocin",
    "sequence": "CYIQNCPLG",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "L",
    "note": "no hemolysis @100 µg/mL, 임상 안전 확인",
    "src": "https://www.researchgate.net/publication/298912862"
  },
  {
    "name": "Pleurocidin",
    "sequence": "GWGSFFKAAAHVGKHVGKAALTHYL",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "L",
    "note": ">256 µg/mL 비용혈, 정밀 HC50 미확인",
    "src": "https://www.sciencedirect.com/science/article/abs/pii/S0882401018316231"
  },
  {
    "name": "Pyrrhocoricin",
    "sequence": "VDKGSYLPRPTPPRPIYNRN",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "L",
    "note": ">600 µg/mL (no hemolysis), non-lytic 메커니즘",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7736402/"
  },
  {
    "name": "GS14Kd4 (gramicidin S analog)",
    "sequence": "cyclic-14aa",
    "hc50_ugml": 125.0,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "D",
    "note": "사이클릭 펩타이드, D-Lys 치환으로 독성 저감 (pepADMET 입력 불가)",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/"
  },
  {
    "name": "D-Piscidin 1 G8P",
    "sequence": "FFHHIFRPIVHVGKTIHRLVTG",
    "hc50_ugml": null,
    "hc50_uM": 8.0,
    "hemolytic": true,
    "aa_type": "D",
    "note": "G8P 치환; 18h 37°C hRBC; 4.4-fold reduction vs D-Piscidin 1",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-Piscidin 1 V12K",
    "sequence": "FFHHIFRGIVHKGKTIHRLVTG",
    "hc50_ugml": null,
    "hc50_uM": 35.0,
    "hemolytic": true,
    "aa_type": "D",
    "note": "V12K 치환; 18h 37°C hRBC; 19.4-fold reduction vs D-Piscidin 1",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-Piscidin 1 G13K",
    "sequence": "FFHHIFRGIVHVKKTIHRLVTG",
    "hc50_ugml": null,
    "hc50_uM": 7.0,
    "hemolytic": true,
    "aa_type": "D",
    "note": "G13K 치환; 18h 37°C hRBC; 3.9-fold reduction vs D-Piscidin 1",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-Dermaseptin S4 L7K",
    "sequence": "ALWMTLKKKVLKAAAKALNAVLVGANA",
    "hc50_ugml": null,
    "hc50_uM": 8.6,
    "hemolytic": true,
    "aa_type": "D",
    "note": "L7K 치환; 18h 37°C hRBC; 14.3-fold reduction vs D-Dermaseptin S4",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-Dermaseptin S4 A14K",
    "sequence": "ALWMTLLKKVLKAKAKALNAVLVGANA",
    "hc50_ugml": null,
    "hc50_uM": 7.0,
    "hemolytic": true,
    "aa_type": "D",
    "note": "A14K 치환; 18h 37°C hRBC; 11.7-fold reduction vs D-Dermaseptin S4",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-RR4",
    "sequence": "WLRRIKAWLRRIKA",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "aa_type": "D",
    "note": "All D-amino acids, C-term amide; HC50 ≥256 µM (최대 시험 농도 미도달); human RBC",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5537347/"
  },
  {
    "name": "DGL13K",
    "sequence": "GKIIKLKASLKLL",
    "hc50_ugml": 1000.0,
    "hc50_uM": 702.0,
    "hemolytic": false,
    "aa_type": "D",
    "note": "All D-amino acids, C-term amide; 1% hRBC, 1h 37°C; dose-response LD50 ~1 mg/mL; MW=1423.87 Da로 µM 환산",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6508730/"
  },
  {
    "name": "TPW-1",
    "sequence": "AGWLLGDINLDALAALAKKIL",
    "hc50_ugml": null,
    "hc50_uM": 45.0,
    "hemolytic": true,
    "aa_type": "MIXED",
    "note": "D-아미노산 부분 치환(D-Asn, D-Asp 포함); P50 측정 vs sheep erythrocyte (human RBC 아님); phosphate buffer pH 7.5",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3584441/"
  },
  {
    "name": "TPW-3",
    "sequence": "AGWLLGDINLKALAALAKKIL",
    "hc50_ugml": null,
    "hc50_uM": 50.0,
    "hemolytic": true,
    "aa_type": "MIXED",
    "note": "D-아미노산 부분 치환(D-Asn 포함); P50 측정 vs sheep erythrocyte (human RBC 아님); phosphate buffer pH 7.5",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3584441/"
  },
  {
    "name": "Protopolybia-MPIII",
    "sequence": "INWLKLGKAVIDAL",
    "hc50_ugml": null,
    "hc50_uM": 23.0,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 23.0±1.2 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Polybia-MPIII",
    "sequence": "IDWLKLGKMVMDVL",
    "hc50_ugml": null,
    "hc50_uM": 38.5,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 38.5±5.9 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Pm-R3",
    "sequence": "INWLKLGKQILGAL",
    "hc50_ugml": null,
    "hc50_uM": 32.3,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 32.3±1.7 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "PDD-B",
    "sequence": "INWLKLGKKILGAL",
    "hc50_ugml": null,
    "hc50_uM": 48.5,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 48.5±3.4 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Agelaia-MPII",
    "sequence": "INWKAILQRIKKML",
    "hc50_ugml": null,
    "hc50_uM": 44.8,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 44.8±10.4 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-J",
    "sequence": "VDWKKIGQHILSVL",
    "hc50_ugml": null,
    "hc50_uM": 69.5,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 69.5±3.2 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "PMM2",
    "sequence": "INWKKIASIGKEVLKAL",
    "hc50_ugml": null,
    "hc50_uM": 42.6,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 42.6±2.5 µM, C-term amide; 17aa; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-Lew",
    "sequence": "INLKALAALAKKIL",
    "hc50_ugml": null,
    "hc50_uM": 82.9,
    "hemolytic": true,
    "aa_type": "L",
    "note": "Mastoparan (L-form); hRBC EC50 82.9±3.8 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Protonectarina-MP",
    "sequence": "INWKALLDAAKKVL",
    "hc50_ugml": null,
    "hc50_uM": 85.2,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 85.2±5.9 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Pm-R1",
    "sequence": "INWLKLGKKILGAI",
    "hc50_ugml": null,
    "hc50_uM": 72.0,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 72.0±8.5 µM, C-term amide; HHA 분류",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-T3",
    "sequence": "INLRGFAALVKKFL",
    "hc50_ugml": null,
    "hc50_uM": 112.1,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 112.1±8.0 µM, C-term amide; MHA 분류 (100-400 µM 범위)",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-II",
    "sequence": "INLKALAALVKKVL",
    "hc50_ugml": null,
    "hc50_uM": 134.6,
    "hemolytic": true,
    "aa_type": "L",
    "note": "hRBC EC50 134.6±1.2 µM, C-term amide; MHA 분류 (100-400 µM 범위)",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "D1",
    "sequence": "KWKSFLKTFKSAKKTVLHTALKAISS",
    "hc50_ugml": 421.5,
    "hc50_uM": 141.0,
    "hemolytic": false,
    "aa_type": "D",
    "note": "all-D enantiomer; Ac-NH2/CONH2 말단; HC50 Table 3; 1% hRBC, 37°C, 18h; MW≈2990 Da; µM=421.5×1000/2990",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3263701/"
  },
  {
    "name": "D2",
    "sequence": "KWKSFLKTFKSAKKTVLHTLLKAISS",
    "hc50_ugml": 83.0,
    "hc50_uM": 27.4,
    "hemolytic": true,
    "aa_type": "D",
    "note": "all-D enantiomer; Ac-NH2/CONH2 말단; HC50 Table 3; 1% hRBC, 37°C, 18h; D1+A20L; MW≈3033 Da",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3263701/"
  },
  {
    "name": "D3",
    "sequence": "KWKSFLKTFKSLKKTVLHTLLKAISS",
    "hc50_ugml": 14.0,
    "hc50_uM": 4.6,
    "hemolytic": true,
    "aa_type": "D",
    "note": "all-D enantiomer; Ac-NH2/CONH2 말단; HC50 Table 3; 1% hRBC, 37°C, 18h; D2+A12L; MW≈3075 Da",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3263701/"
  },
  {
    "name": "D4",
    "sequence": "KWKSFLKTFKSLKKTVLHTLLKLISS",
    "hc50_ugml": 3.5,
    "hc50_uM": 1.1,
    "hemolytic": true,
    "aa_type": "D",
    "note": "all-D enantiomer; Ac-NH2/CONH2 말단; HC50 Table 3; 1% hRBC, 37°C, 18h; D3+A23L; MW≈3117 Da; 최고 용혈성",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3263701/"
  },
  {
    "name": "D5",
    "sequence": "KWKSFLKTFKSLKKTKLHTLLKLISS",
    "hc50_ugml": 47.0,
    "hc50_uM": 14.9,
    "hemolytic": true,
    "aa_type": "D",
    "note": "all-D enantiomer; Ac-NH2/CONH2 말단; HC50 Table 3; 1% hRBC, 37°C, 18h; D4+V16K; MW≈3146 Da",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3263701/"
  }
]
```

---

## 통계 요약

| 분류 | 개수 |
|------|------|
| 전체 항목 | 52 |
| 유효 서열 (표준 20AA) | 50 |
| 사이클릭 (제외) | 2 |
| L-aa | 34 |
| D-aa | 16 (+ 사이클릭 1 = 총 17 D-aa labeled) |
| MIXED | 2 |
| hemolytic=true | 38 |
| hemolytic=false | 14 |

## D-aa 항목 목록

| # | 이름 | 서열 | HC50 | hemolytic | 출처 |
|---|------|------|------|-----------|------|
| 1 | D-Piscidin 1 | FFHHIFRGIVHVGKTIHRLVTG | 1.8 µM | true | PMC4014698 |
| 2 | D-Piscidin 1 I9K | FFHHIFRGKVHVGKTIHRLVTG | 98 µM | false | PMC4014698 |
| 3 | D-Piscidin 1 G8P | FFHHIFRPIVHVGKTIHRLVTG | 8 µM | true | PMC4014698 |
| 4 | D-Piscidin 1 V12K | FFHHIFRGIVHKGKTIHRLVTG | 35 µM | true | PMC4014698 |
| 5 | D-Piscidin 1 G13K | FFHHIFRGIVHVKKTIHRLVTG | 7 µM | true | PMC4014698 |
| 6 | D-Dermaseptin S4 | ALWMTLLKKVLKAAAKALNAVLVGANA | 0.6 µM | true | PMC4014698 |
| 7 | D-Dermaseptin S4 L7K | ALWMTLKKKVLKAAAKALNAVLVGANA | 8.6 µM | true | PMC4014698 |
| 8 | D-Dermaseptin S4 A14K | ALWMTLLKKVLKAKAKALNAVLVGANA | 7 µM | true | PMC4014698 |
| 9 | D-Dermaseptin S4 L7K,A14K | ALWMTLKKKVLKAKAAKALNAVLVGANA | 241 µM | false | PMC4014698 |
| 10 | D-RR4 | WLRRIKAWLRRIKA | ≥256 µM | false | PMC5537347 |
| 11 | DGL13K | GKIIKLKASLKLL | ~1000 µg/mL | false | PMC6508730 |
| 12 | D1 | KWKSFLKTFKSAKKTVLHTALKAISS | 421.5 µg/mL | false | PMC3263701 |
| 13 | D2 | KWKSFLKTFKSAKKTVLHTLLKAISS | 83 µg/mL | true | PMC3263701 |
| 14 | D3 | KWKSFLKTFKSLKKTVLHTLLKAISS | 14 µg/mL | true | PMC3263701 |
| 15 | D4 | KWKSFLKTFKSLKKTVLHTLLKLISS | 3.5 µg/mL | true | PMC3263701 |
| 16 | D5 | KWKSFLKTFKSLKKTKLHTLLKLISS | 47 µg/mL | true | PMC3263701 |
| MIXED-1 | TPW-1 | AGWLLGDINLDALAALAKKIL | 45 µM | true | PMC3584441 |
| MIXED-2 | TPW-3 | AGWLLGDINLKALAALAKKIL | 50 µM | true | PMC3584441 |

## §검증 필요

1. **D-Dermaseptin S4 A14K 서열**: PMC4014698 Table에서 ALWMTLLKKVLKAKAKALNAVLVGANA로 추출. A14K 변이(pos14 A→K) 적용 결과 ✓
2. **TPW-1/TPW-3 RBC 종류**: Sheep erythrocyte 사용 — human RBC 비교 시 약 1.5-2× 차이 가능성. note에 명시됨.
3. **DGL13K hc50_uM**: ~1000 µg/mL ÷ 1.42387 g/mmol ≈ 702 µM. 1h 인큐베이션 (다른 항목 18h). 과소평가 가능.
4. **Mastoparan-T3/II (MHA 분류)**: EC50 > 100 µM이나 measurable hemolysis → hemolytic=true로 처리. pepADMET가 이를 과소평가할 가능성 높음.
5. **D1~D5 서열 (PMC3263701)**: Table 1에서 직접 추출 확인. Ac-NH2/CONH2 말단 수정이 있으나 pepADMET 입력 시 표준 서열만 사용. HC50은 Table 3에서 직접 측정값 확인 ✓. D1~D5 모두 all-D enantiomers.
6. **D1 hemolytic 분류**: 421.5 µg/mL — 50% hemolysis 달성되나 매우 고농도(>200 µg/mL threshold → false로 분류). 기존 DGL13K(1000 µg/mL, false), Magainin2(200 µg/mL, false)와 동일 기준 적용.
