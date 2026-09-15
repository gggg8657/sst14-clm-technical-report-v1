# BENCH_admet_literature — 용혈(Hemolytic) Ground-Truth 데이터셋

**목적**: pepADMET HC50/독성 예측 검증용 문헌 기반 벤치마크  
**작성일**: 2026-06-17  
**작성자**: researcher (Claude Sonnet 4.6)

---

## 1. 검색 전략

| 쿼리 | 대상 DB/소스 |
|------|------------|
| `melittin hemolytic HC50 peptide sequence measured value µg/mL` | WebSearch |
| `HemoPI database hemolytic peptide dataset benchmark sequences` | WebSearch |
| `DBAASP database hemolytic antimicrobial peptide HC50 MHC measured` | WebSearch |
| `magainin 2 cecropin defensin LL-37 HC50 measured sequence` | WebSearch |
| `mastoparan hemolytic EC50 wasp venom PMC10611374` | WebFetch |
| `piscidin 1 dermaseptin S4 HC50 measured table PMC4014698` | WebFetch |
| `non-hemolytic antimicrobial peptide indolicidin temporin mastoparan` | WebSearch |
| `proline-rich AMP apidaecin non-hemolytic` | WebSearch |
| `aurein 1.2 GLFDIIKKIAESF hemolytic HC50` | WebSearch |
| `oxytocin vasopressin non-hemolytic clinical peptide` | WebSearch |
| `gramicidin S cyclic peptide HC50 measured` | WebSearch + WebFetch |
| HemoPI 논문 (PMC4782144) | WebFetch |
| Hemolytik 논문 (NAR 2014) | WebFetch |
| Communications Biology 2025 (HC50 예측) | WebSearch |

---

## 2. 발견 자료 목록

1. Suttmann H et al. "The current landscape of the antimicrobial peptide melittin and its therapeutic potential" *Front. Immunol.* 2024 — PMC10838977
2. Savoia D et al. HemoPI — "A web server and mobile app for computing hemolytic potency of peptides" *Sci. Rep.* 2016;6:22843 — PMC4782144
3. Gautam A et al. "Hemolytik: a database of experimentally determined hemolytic and non-hemolytic peptides" *Nucleic Acids Res.* 2014;42:D444 — DOI:10.1093/nar/gkt1008
4. Chung CR et al. "Prediction of hemolytic peptides and their hemolytic concentration" *Commun. Biol.* 2025;8 — DOI:10.1038/s42003-025-07615-w
5. Saravanan R et al. "Specificity Determinants Improve Therapeutic Indices of Piscidin 1 and Dermaseptin S4" *Pharmaceuticals* 2014;7:366 — PMC4014698
6. Sani M-A, Separovic F. "Characterization of the Hemolytic Activity of Mastoparan Family Peptides from Wasp Venoms" *Toxins* 2023;15:591 — PMC10611374
7. Kondejewski LH et al. "Effects of single d-amino acid substitutions on disruption of β-sheet structure and hydrophobicity in cyclic 14-residue antimicrobial peptide analogs related to gramicidin S" *Antimicrob. Agents Chemother.* 2006 — PMC1464084
8. Scocchi M et al. "Traditional and Computational Screening of Non-Toxic Peptides" *Int. J. Mol. Sci.* 2022 — PMC8953747
9. Pandey BK et al. "Correlation between hemolytic activity, cytotoxicity and systemic in vivo toxicity of synthetic AMPs" *Sci. Rep.* 2020 — DOI:10.1038/s41598-020-69995-9
10. Rozek T et al. "The antibiotic and anticancer active aurein peptides from Australian Bell Frogs" *Eur. J. Biochem.* 2000;267:5330 — DOI:10.1046/j.1432-1327.2000.01536.x

---

## 3. 펩타이드 벤치마크 표

> **범례**: HC50 = 50% 용혈 농도. MHC = 최소 용혈 농도(일부 연구). EC50 = 50% 효과 농도(mastoparan 계열). 
> 값이 없는 칸은 출처에서 수치를 확인하지 못한 경우 (null).
> ⚠️ pepADMET 학습셋 중복 위험: DBAASP/HemoPI 기반 데이터는 pepADMET 학습에 사용되었을 가능성 있음.

### 독성 (Hemolytic) 펩타이드

| name | sequence (1-letter) | HC50 / EC50 (단위) | class | 출처 URL |
|------|--------------------|--------------------|-------|----------|
| Melittin | GIGAVLKVLTTGLPALISWIKRKRQQ | 16.28 µg/mL (2% hRBC) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10838977/ |
| Gramicidin S (native) | cyclo(VKLKVL)₂ — 10aa 사이클릭 | ~16 µg/mL (합성품) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/ |
| GS14Ll4 (14aa 유사체) | cyclic 14aa (Leu at pos4) | 3 µg/mL | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/ |
| GS14Fd4 | cyclic 14aa (D-Phe at pos4) | 2 µg/mL | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/ |
| GS14G4 | cyclic 14aa (Gly at pos4) | 3 µg/mL | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/ |
| D-Piscidin 1 | FFHHIFRGIVHVGKTIHRLVTG | 1.8 µM (18h 37°C) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/ |
| D-Dermaseptin S4 | ALWMTLLKKVLKAAAKALNAVLVGANA | 0.6 µM (18h 37°C) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/ |
| Agelaia-MPI | INWLKLGKAIIDAL-NH2 | 3.7 µM (hRBC) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/ |
| Mastoparan-T2 | INLKVFAALVKKLL-NH2 | 11.8 µM (hRBC) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/ |
| Mastoparan-T1 | INLKVFAALVKKFL-NH2 | 30.8 µM (hRBC) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/ |
| Mastoparan-C | INLKALLAVAKKIL-NH2 | 30.2 µM (hRBC) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/ |
| Polybia-MPII | INWLKLGKMVIDAL-NH2 | 23.3 µM (hRBC) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/ |
| Mastoparan-T4 | INLFGFAALVKKFL-NH2 | 28.5 µM (hRBC) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/ |
| Aurein 1.2 | GLFDIIKKIAESF-NH2 | ~30 µM (horse/human erythrocyte) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC6695355/ |
| Indolicidin | ILPWKWPWWPWRR-NH2 | 측정값 확인 불가 (확실히 용혈성) | hemolytic | https://pubmed.ncbi.nlm.nih.gov/8849687/ |
| MG-H1 (Magainin 유도체) | — | 2.9 µM | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC8953747/ |
| PGLa | GMASKAGAIAGKIAKVALKAL-NH2 | 600 nM | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC8953747/ |
| EpVP2b | FDLLGLVKSVVSAL-NH2 | 34.1 µM (hRBC) | hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/ |

### 비독성 / 저독성 (Non-hemolytic / Low-hemolytic) 펩타이드

| name | sequence (1-letter) | HC50 / MHC (단위) | class | 출처 URL |
|------|--------------------|--------------------|-------|----------|
| Magainin 2 | GIGKFLHSAKKFGKAFVGEIMNS | >200 µg/mL (≈250 µg/mL 이상) | non-hemolytic | https://www.sciencedirect.com/topics/medicine-and-dentistry/magainin-2 |
| Cecropin A | KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK | >169 µg/mL (또는 <5% 용혈 @800 µM) | non-hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC10940386/ |
| LL-37 | LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES | ~8% lysis @20 µM (HC50 도달 안 함, >200 µM 추정) | low-hemolytic | https://www.sciencedirect.com/topics/chemistry/ll-37 |
| Buforin II | TRSSRAGLQFPVGRVHRLLRK | 측정값 없음 (negligible hemolysis) | non-hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC6801614/ |
| Dermaseptin S1 | ALWKTMLKKLGTMALHAGKAALGAAADTISQGTQ | non-hemolytic (HC50 미도달 확인) | non-hemolytic | https://onlinelibrary.wiley.com/doi/full/10.1111/j.1600-0463.2010.02637.x |
| D-Dermaseptin S4 L7K,A14K | ALWMTLKKKVLKAKAAKALNAVLVGANA | 241 µM (18h) | non-hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/ |
| D-Piscidin 1 I9K | FFHHIFRGKVHVGKTIHRLVTG | 98 µM | low-hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/ |
| GS14Kd4 | cyclic 14aa (D-Lys at pos4) | 125 µg/mL | low-hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/ |
| GS14Nd4 | cyclic 14aa (D-Asn at pos4) | 62 µg/mL | low-hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/ |
| Oxytocin | CYIQNCPLG-NH2 | >100 µg/mL (no detectable hemolysis) | non-hemolytic | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2423826/ |
| Pleurocidin | GWGSFFKAAAHVGKHVGKAALTHYL | >256 µg/mL (non-hemolytic 보고) | non-hemolytic | https://www.sciencedirect.com/science/article/abs/pii/S0882401018316231 |
| Pyrrhocoricin (proline-rich AMP) | VDKGSYLPRPTPPRPIYNRN | >600 µg/mL (no hemolysis) | non-hemolytic | https://pmc.ncbi.nlm.nih.gov/articles/PMC7736402/ |
| Ropalidia-MP | INWAKLGKLALQAL-NH2 | 42.5 µM (hRBC) | borderline | https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/ |

---

## 4. 비교 표 — 방법론·범위·데이터 출처 차이

| 항목 | DBAASP v3 | HemoPI / Hemolytik | 개별 실험 논문 |
|------|-----------|-------------------|--------------|
| 데이터 규모 | HC50 수록 3,147종 | 2,970~13,215개 항목 | 수십~수백 |
| 측정 단위 | µg/mL or µM 혼재 | µM 주 | 논문마다 상이 |
| RBC 출처 | 인간/동물 혼재 | 1~17가지 | 명시 필요 |
| 인큐베이션 시간 | 다양 | 18h 37°C 기준 | 1h / 18h 결과 차이 최대 7× |
| 학습셋 사용 | pepADMET 가능성 높음 | pepADMET 가능성 높음 | 독립적 가능성 상대적으로 높음 |

**주요 방법론 차이**: 1h 인큐베이션 vs. 18h 인큐베이션에서 HC50 값이 최대 7배 차이날 수 있음 (D-Piscidin 1 G8P: 55 µM @1h → 8 µM @18h, PMC4014698).

---

## 5. 본 프로젝트 적용 가능성

| 펩타이드 그룹 | 적용 가능성 | 비고 |
|------------|----------|------|
| Melittin (강 양성 대조군) | **HIGH** | HC50 16.28 µg/mL, 출처 명확, pepADMET 벤치마크 표준 |
| D-Piscidin 1 / D-Dermaseptin S4 (용혈성) | **HIGH** | µM 단위 측정값, 인큐베이션 조건 명확 |
| Mastoparan 계열 5종 | **HIGH** | hRBC EC50 복수 측정, 독성 분포 폭넓음 |
| Magainin 2, Cecropin A (저독성 대조군) | **HIGH** | 비용혈 레퍼런스로 광범위 검증됨 |
| LL-37 | **MED** | 수치 HC50 미확인, 추정값 사용 |
| Oxytocin (임상 비독성) | **HIGH** | 명확한 비용혈 증거, 임상 안전성 뒷받침 |
| Pleurocidin, Pyrrhocoricin | **MED** | >256 µg/mL 보고이나 정밀 HC50 미확인 |
| GS14 유사체 계열 | **MED** | 사이클릭 펩타이드 — pepADMET 선형 모델 외삽 주의 |
| Buforin II, Dermaseptin S1 | **MED** | 비용혈 확인되나 정량값 부재 |
| Aurein 1.2 | **MED** | ~30 µM이나 측정 조건 불명확 |

---

## 6. §검증 필요

1. **LL-37 정밀 HC50**: 검색 결과에서 수치 미확인. Hancock 그룹 원논문 (FEBS 2002) 또는 DBAASP entry 직접 접근 필요.
2. **Indolicidin HC50 수치**: "확실히 용혈성"이나 µg/mL 수치 미확보. PubMed PMID:8849687 원문 확인 필요.
3. **인큐베이션 조건 통일**: 1h vs. 18h 결과 차이 고려 — pepADMET가 어떤 조건에서 학습되었는지 확인 필요.
4. **pepADMET 학습셋 중복**: DBAASP/HemoPI 기반 데이터는 pepADMET 학습셋에 포함 가능성이 높음. 가능하면 개별 논문에서 직접 인용된 데이터(아래 ★) 우선 사용 권장.
   - ★ 독립성 상대적으로 높은 소스: PMC4014698 (piscidin/dermaseptin 실측), PMC10611374 (mastoparan EC50), PMC1464084 (gramicidin S 유사체)
5. **사이클릭 펩타이드 (Gramicidin S, GS14 유사체)**: 선형 서열 표기로는 pepADMET 입력 불가 — 검증 시 주의 또는 제외 권고.
6. **µg/mL ↔ µM 변환**: 분자량 없이 단위 환산 불가한 항목 다수. 리뷰어가 변환 전 MW 확인 필요.
7. **PGLa 수치 재확인**: 600 nM은 비교적 강한 용혈성이나 원논문 확인 필요 (PMC8953747 인용 소스).

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
    "note": "2% hRBC, 측정값 ±0.17",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10838977/"
  },
  {
    "name": "D-Piscidin 1",
    "sequence": "FFHHIFRGIVHVGKTIHRLVTG",
    "hc50_ugml": null,
    "hc50_uM": 1.8,
    "hemolytic": true,
    "note": "18h 37°C hRBC",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-Dermaseptin S4",
    "sequence": "ALWMTLLKKVLKAAAKALNAVLVGANA",
    "hc50_ugml": null,
    "hc50_uM": 0.6,
    "hemolytic": true,
    "note": "18h 37°C hRBC",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "Agelaia-MPI",
    "sequence": "INWLKLGKAIIDAL",
    "hc50_ugml": null,
    "hc50_uM": 3.7,
    "hemolytic": true,
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-T2",
    "sequence": "INLKVFAALVKKLL",
    "hc50_ugml": null,
    "hc50_uM": 11.8,
    "hemolytic": true,
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Polybia-MPII",
    "sequence": "INWLKLGKMVIDAL",
    "hc50_ugml": null,
    "hc50_uM": 23.3,
    "hemolytic": true,
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-T4",
    "sequence": "INLFGFAALVKKFL",
    "hc50_ugml": null,
    "hc50_uM": 28.5,
    "hemolytic": true,
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-C",
    "sequence": "INLKALLAVAKKIL",
    "hc50_ugml": null,
    "hc50_uM": 30.2,
    "hemolytic": true,
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Mastoparan-T1",
    "sequence": "INLKVFAALVKKFL",
    "hc50_ugml": null,
    "hc50_uM": 30.8,
    "hemolytic": true,
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "EpVP2b",
    "sequence": "FDLLGLVKSVVSAL",
    "hc50_ugml": null,
    "hc50_uM": 34.1,
    "hemolytic": true,
    "note": "hRBC EC50, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Ropalidia-MP",
    "sequence": "INWAKLGKLALQAL",
    "hc50_ugml": null,
    "hc50_uM": 42.5,
    "hemolytic": true,
    "note": "hRBC EC50, borderline, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10611374/"
  },
  {
    "name": "Aurein 1.2",
    "sequence": "GLFDIIKKIAESF",
    "hc50_ugml": null,
    "hc50_uM": 30.0,
    "hemolytic": true,
    "note": "~30 µM horse/human erythrocyte, C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6695355/"
  },
  {
    "name": "PGLa",
    "sequence": "GMASKAGAIAGKIAKVALKAL",
    "hc50_ugml": null,
    "hc50_uM": 0.6,
    "hemolytic": true,
    "note": "HC50 600 nM (0.6 µM), C-term amide",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8953747/"
  },
  {
    "name": "Indolicidin",
    "sequence": "ILPWKWPWWPWRR",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": true,
    "note": "수치 미확인, 강한 용혈성 확인됨, C-term amide",
    "src": "https://pubmed.ncbi.nlm.nih.gov/8849687/"
  },
  {
    "name": "GS14Ll4 (gramicidin S analog)",
    "sequence": "cyclic-14aa",
    "hc50_ugml": 3.0,
    "hc50_uM": null,
    "hemolytic": true,
    "note": "사이클릭 펩타이드, 선형 서열 표기 불가",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/"
  },
  {
    "name": "Magainin 2",
    "sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
    "hc50_ugml": 200.0,
    "hc50_uM": null,
    "hemolytic": false,
    "note": "HC50 ~200-500 µg/mL, therapeutic index 10-20",
    "src": "https://www.sciencedirect.com/topics/medicine-and-dentistry/magainin-2"
  },
  {
    "name": "Cecropin A",
    "sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "hc50_ugml": 169.0,
    "hc50_uM": null,
    "hemolytic": false,
    "note": "HC50 169 µg/mL 또는 <5% 용혈 @800 µM",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10940386/"
  },
  {
    "name": "LL-37",
    "sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "note": "~8% lysis @20 µM, HC50 >200 µM 추정, 수치 확인 필요",
    "src": "https://www.sciencedirect.com/topics/chemistry/ll-37"
  },
  {
    "name": "Buforin II",
    "sequence": "TRSSRAGLQFPVGRVHRLLRK",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "note": "negligible hemolysis 확인, 정량값 없음",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6801614/"
  },
  {
    "name": "Dermaseptin S1",
    "sequence": "ALWKTMLKKLGTMALHAGKAALGAAADTISQGTQ",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "note": "non-hemolytic but fungicidal, HC50 미도달",
    "src": "https://onlinelibrary.wiley.com/doi/full/10.1111/j.1600-0463.2010.02637.x"
  },
  {
    "name": "D-Piscidin 1 I9K",
    "sequence": "FFHHIFRGKVHVGKTIHRLVTG",
    "hc50_ugml": null,
    "hc50_uM": 98.0,
    "hemolytic": false,
    "note": "54-fold reduction vs parent, 18h 37°C",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "D-Dermaseptin S4 L7K,A14K",
    "sequence": "ALWMTLKKKVLKAKAAKALNAVLVGANA",
    "hc50_ugml": null,
    "hc50_uM": 241.0,
    "hemolytic": false,
    "note": "401-fold reduction vs parent, 18h 37°C",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4014698/"
  },
  {
    "name": "Oxytocin",
    "sequence": "CYIQNCPLG",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "note": "no hemolysis @100 µg/mL, 임상 안전 확인",
    "src": "https://www.researchgate.net/publication/298912862"
  },
  {
    "name": "Pleurocidin",
    "sequence": "GWGSFFKAAAHVGKHVGKAALTHYL",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "note": ">256 µg/mL 비용혈, 정밀 HC50 미확인",
    "src": "https://www.sciencedirect.com/science/article/abs/pii/S0882401018316231"
  },
  {
    "name": "Pyrrhocoricin (proline-rich AMP)",
    "sequence": "VDKGSYLPRPTPPRPIYNRN",
    "hc50_ugml": null,
    "hc50_uM": null,
    "hemolytic": false,
    "note": ">600 µg/mL (no hemolysis), non-lytic 메커니즘",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7736402/"
  },
  {
    "name": "GS14Kd4 (gramicidin S analog)",
    "sequence": "cyclic-14aa",
    "hc50_ugml": 125.0,
    "hc50_uM": null,
    "hemolytic": false,
    "note": "사이클릭 펩타이드, D-Lys 치환으로 독성 저감",
    "src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1464084/"
  }
]
```

---

## 참고: 주요 DB 정보

- **DBAASP v3** (2021): https://dbaasp.org — HC50 수록 3,147종, 총 15,700+ 항목
- **Hemolytik** (2014/업데이트): http://crdd.osdd.net/raghava/hemolytik/ — 2,970~13,215 항목
- **HemoPI** (2016): 552 hemolytic + 552 non-hemolytic (HemoPI-1 benchmark)
- **AmpLyze/HemoPI2 학습셋**: 1,926 펩타이드 with measured HC50 (Chung et al. 2025, Commun. Biol.)

⚠️ **학습셋 중복 경고**: 위 DB들은 pepADMET의 훈련 데이터 소스일 가능성이 있어 벤치마크 독립성이 보장되지 않음. 독립적 검증에는 개별 논문 실험 데이터(PMC4014698, PMC10611374, PMC10838977 등)를 우선 사용할 것.
