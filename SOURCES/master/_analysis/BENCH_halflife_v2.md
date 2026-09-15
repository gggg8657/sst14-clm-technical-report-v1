# 펩타이드 반감기 벤치마크 데이터셋 v2

**작성일**: 2026-06-17  
**작성자**: researcher (PRST_N_FM)  
**목적**: Spearman 상관계수 안정화를 위한 N 확장 및 L-aa군 vs D-aa 포함군 분리 비교

---

## 1. 검색 전략

| 쿼리 유형 | 주요 키워드 |
|-----------|------------|
| L-aa 군 | glucagon, calcitonin, ANP, BNP, secretin, VIP, ghrelin, GLP-1, PYY, oxytocin, vasopressin, somatostatin-14, substance P, bradykinin, angiotensin II, enkephalin, insulin, neurotensin |
| D-aa 군 | octreotide, lanreotide, vapreotide, pasireotide, DOTATATE/DOTATOC, leuprolide, goserelin, buserelin, triptorelin, cetrorelix, degarelix, icatibant, desmopressin, bivalirudin |
| 데이터베이스 | PubMed, DrugBank, FDA label (DailyMed), Wikipedia pharmacokinetics, PeptideJournal.org |

**주의**: 모든 t½ 값은 인간(human)에서의 혈청/혈장 데이터를 우선. 불가시 동물 데이터 명기. 최종 단위는 **시간(h)** 환산. 출처 URL 필수.

---

## 2. 데이터 표

### 2-1. L-아미노산 군 (표준, 비변형 or 천연/재조합)

| # | Name | Sequence (1-letter) | Half-life (h) | aa_type | Modifications | Source URL | Note |
|---|------|---------------------|---------------|---------|---------------|------------|------|
| 1 | Somatostatin-14 (SST-14) | AGCKNFFWKTFTSC | 0.033 (2 min) | L | cyclization (Cys3-Cys14 SS bond) | [StatPearls NBK538327](https://www.ncbi.nlm.nih.gov/books/NBK538327/) | IV 투여 기준; 순환내 50% 제거 < 3 min |
| 2 | Glucagon | HSQGTFTSDYSKYLDSRRAQDFVQWLMNT | 0.17 (10 min) | L | none | [PubMed 8157042](https://pubmed.ncbi.nlm.nih.gov/8157042/) | IV 볼루스 인간 t½ 8–18 min; 중간값 ~10 min 채택 |
| 3 | Insulin (native) | GIVEQCCTSICSLYQLENYCN / FVNQHLCGSHLVEALYLVCGERGFFYTPKT | 0.10 (6 min) | L | disulfide bonds (intrachain+interchain) | [Drugs.com insulin](https://www.drugs.com/monograph/vasopressin.html) | 문헌값 5–6 min; 수용성 단일 t½ 기준 |
| 4 | Oxytocin | CYIQNCPLG (amide, Cys1-Cys6 SS) | 0.067 (4 min) | L | cyclization (disulfide), C-term amide | [Wikipedia Oxytocin](https://en.wikipedia.org/wiki/Oxytocin) | IV 기준 1–6 min; 중간값 ~4 min |
| 5 | Vasopressin (ADH) | CYFQNCPRG (amide, Cys1-Cys6 SS) | 0.25 (15 min) | L | cyclization (disulfide), C-term amide | [PeptideJournal.org](https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference) | IV 10–20 min; 중간값 15 min |
| 6 | ANP (atrial natriuretic peptide) | SLRRSSCFGGRMDRIGAQSGLGCNSFRY (28aa, Cys7-Cys23 SS) | 0.050 (3 min) | L | cyclization (disulfide) | [Acad Cardiovasc Res](https://academic.oup.com/cardiovascres/article/51/3/442/367519) | 인간 혈장 t½ 2–5 min; 3 min 채택 |
| 7 | BNP (brain natriuretic peptide) | SPKMVQGSGCFGRKMDRISSSSGLGCKVLRRH (32aa, Cys10-Cys26 SS) | 0.33 (20 min) | L | cyclization (disulfide) | [PMC 9312360](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9312360/) | 인간 혈장 t½ 20 min 확립값 |
| 8 | Nesiritide (recombinant BNP) | SPKMVQGSGCFGRKMDRISSSSGLGCKVLRRH | 0.30 (18 min) | L | none (recombinant, no modification) | [FDA label Natrecor](https://www.accessdata.fda.gov/drugsatfda_docs/label/2009/020920s023lbl.pdf) | 인간 심부전 환자 terminal t½ 18 min |
| 9 | Secretin (endogenous) | HSDGTFTSELSRLREGARLQRLLQGLV | 0.067 (4 min) | L | none | [PubMed 830238](https://pubmed.ncbi.nlm.nih.gov/830238/) | RIA 측정 인간 혈장 t½ 4.06 ± 0.82 min |
| 10 | VIP (vasoactive intestinal peptide) | HSDAVFTDNYTRLRKQMAVKKYLNSILN | 0.025 (1.5 min) | L | none | [PubMed 730072](https://pubmed.ncbi.nlm.nih.gov/730072/) | 인간 IV 주입 중단 후 소실 t½ ~1 min; 1.5 min 채택 |
| 11 | GLP-1 (7-37, native) | HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR | 0.033 (2 min) | L | none | [PMC 12052016](https://pmc.ncbi.nlm.nih.gov/articles/PMC12052016/) | DPP-4 신속 분해; ~2 min |
| 12 | GIP (native) | YAEGTFISDYSIAMDKIHQQDFVNWLLAQK | 0.083 (5 min) | L | none | [Nature Nutr Diabetes 2025](https://www.nature.com/articles/s41387-025-00397-4) | ~5 min 내피 DPP-4 분해 |
| 13 | PYY(3-36) | IKPEAPGEDASPEELNRYYASLRHYLNLVTRQRY (33aa, pos 3–36) | 0.060 (3.6 min) | L | none | [PMC 4552532](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4552532/) | 마우스 in vivo t½ 3.6 ± 0.5 min; 인간 유사값 적용 |
| 14 | Ghrelin (acylated) | GSSFLSPEHQRVQQRKESKKPPAKLQPR (28aa; Ser3 octanoyl) | 0.183 (11 min) | L | fatty_acid (Ser3-octanoyl) | [EJE 168:6 2013](https://eje.bioscientifica.com/view/journals/eje/168/6/821.xml) | 활성형(acyl-ghrelin) t½ 9–13 min; 중간값 11 min. Ser3 acylation은 활성에 필수이나 변형으로 분류 |
| 15 | Substance P | RPKPQQFFGLM | 0.017 (1 min) | L | none | [PeptideJournal.org](https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference) | in vitro 혈장 ~1–2 min; 1 min 채택 |
| 16 | Bradykinin | RPPGFSPFR | 0.0028 (10 sec) | L | none | [ScienceDirect Bradykinin](https://www.sciencedirect.com/science/article/abs/pii/000629529190106F) | IV 기준 수초(seconds) 내 제거; 10 sec(0.0028 h) 채택 |
| 17 | Angiotensin II | DRVYIHPF | 0.0083 (0.5 min) | L | none | [PubMed 9231819](https://pubmed.ncbi.nlm.nih.gov/9231819/) | 순환 intact Ang II t½ ~0.5 min |
| 18 | Neurotensin | QLYPQRPYIL (13aa: pGlu-Leu-Tyr-Glu-Asn-Lys-Pro-Arg-Arg-Pro-Tyr-Ile-Leu) | 0.025 (1.5 min) | L | none | [ScienceDirect 1986](https://www.sciencedirect.com/science/article/pii/0196978186900549) | 인간 혈장 intact t½ 1.5 min |
| 19 | Met-enkephalin | YGGFM | 0.017 (1 min) | L | none | [Wikipedia Met-enkephalin](https://en.wikipedia.org/wiki/Met-enkephalin) | 인간 혈장 < 2 min; 1 min 채택 |
| 20 | Leu-enkephalin | YGGFL | 0.017 (1 min) | L | none | [PMC 8308721](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8308721/) | 혈장 t½ 1–2 min; 1 min 채택 |
| 21 | GnRH (native) | QHWSYGLRPG | 0.050 (3 min) | L | C-term amide, pGlu N-term | [PMC 5461264](https://pmc.ncbi.nlm.nih.gov/articles/PMC5461264/) | 문헌 2–4 min; 3 min 채택 |
| 22 | Calcitonin (salmon) | CSNLSTCVLGKLSQELHKLQTYPRTNTGSGTP (32aa) | 1.0 h | L | cyclization (Cys1-Cys7 SS, ring) | [Drugs.com Calcitonin](https://www.drugs.com/pro/calcitonin-salmon-injection.html) | SC/IM 투여 terminal t½ 58–64 min; ~1 h 채택 |
| 23 | Exenatide (Byetta) | HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS | 2.4 h | L | none (all L-aa, exendin-4 기반; Gly2가 DPP-4 저항성 부여) | [PubChem CID 45588096](https://pubchem.ncbi.nlm.nih.gov/compound/Exenatide) | 모든 L-aa이지만 Gly2에 의해 DPP-4 저항. aa_type=L로 분류 |
| 24 | Sermorelin (GHRH 1-29) | YADAIFTNSYRKVLGQLSARKLLQDIMSRQQGESNQER (29aa) | 0.183 (11 min) | L | none | [PeptideJournal.org](https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference) | t½ 11–12 min |
| 25 | GHRP-6 | HWAWFK (His-D-Trp-Ala-Trp-D-Phe-Lys) | 0.42 h | mixed | d_amino_acid (D-Trp2, D-Phe5) | [PeptideJournal.org](https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference) | 20–30 min; D-aa 2개 포함 → mixed 분류 |

> **주의 (Exenatide)**: exendin-4 유래이며 모든 위치가 L-아미노산이나 Gly2 치환으로 DPP-4 저항성. aa_type=L 유지.  
> **주의 (GHRP-6)**: D-Trp, D-Phe 포함이므로 L 군이 아닌 mixed로 이동.

---

### 2-2. D-아미노산 포함 군 (치료 펩타이드)

| # | Name | Sequence (1-letter + D-aa 위치 주석) | Half-life (h) | aa_type | Modifications | Source URL | Note |
|---|------|--------------------------------------|---------------|---------|---------------|------------|------|
| 26 | Octreotide | D-Phe¹-Cys²-Phe³-D-Trp⁴-Lys⁵-Thr⁶-Cys⁷-Thr⁸-ol (8aa cyclic) | 1.7 h | mixed | cyclization (Cys2-Cys7 SS), d_amino_acid (D-Phe1, D-Trp4), C-term alcohol | [FDA label Sandostatin](https://www.accessdata.fda.gov/drugsatfda_docs/label/2005/019667s050lbl.pdf) | SC 투여 t½ ~100 min = 1.7 h; IV 이상성 10/90 min |
| 27 | Lanreotide | D-Nal¹-Cys²-Tyr³-D-Trp⁴-Lys⁵-Val⁶-Cys⁷-Thr-NH₂ (8aa cyclic) | 1.5 h (SC, 비depot) | mixed | cyclization (Cys2-Cys7 SS), d_amino_acid (D-Nal1, D-Trp4), C-term amide | [PMC 3269320](https://pmc.ncbi.nlm.nih.gov/articles/PMC3269320/) | 단기 SC t½ 1–2 h; Autogel depot t½ 23–30 d (별도 formulation) |
| 28 | Vapreotide | D-Phe¹-Cys²-Tyr³-D-Trp⁴-Lys⁵-Val⁶-Cys⁷-Trp⁸-NH₂ (8aa cyclic) | 0.50 h (30 min) | mixed | cyclization (Cys2-Cys7 SS), d_amino_acid (D-Phe1, D-Trp4), C-term amide | [ScienceDirect Vapreotide](https://www.sciencedirect.com/topics/medicine-and-dentistry/vapreotide) | t½ 30 min |
| 29 | Pasireotide | 비표준 aa 포함 cyclic hexapeptide (D-Phe-Hyp-Phg-D-Trp-Lys-Tyr) | 12 h | mixed | cyclization, d_amino_acid (D-Phe, D-Trp), non-standard aa (Hyp, Phg) | [PubMed 22364824](https://pubmed.ncbi.nlm.nih.gov/22364824/) | SC 인간 t½ 10–13 h; 중간값 12 h |
| 30 | Leuprolide (leuprorelin) | pGlu-His-Trp-Ser-Tyr-D-Leu⁶-Leu-Arg-Pro-NHEt (9aa + NEt) | 3.0 h | mixed | d_amino_acid (D-Leu6), C-term ethylamide, pGlu N-term | [Wikipedia Leuprorelin](https://en.wikipedia.org/wiki/Leuprorelin) | SC 수용액 t½ ~3 h |
| 31 | Goserelin | pGlu-His-Trp-Ser-Tyr-D-Ser(tBu)⁶-Leu-Arg-Pro-AzGly-NH₂ (10aa) | 3.3 h (남), 2.3 h (여) | mixed | d_amino_acid (D-Ser6, tBu보호), C-term azaGly amide | [PubMed 10926349](https://pubmed.ncbi.nlm.nih.gov/10926349/) | 문헌값 남 4.2 h / 여 2.3 h; 평균 ~3.3 h |
| 32 | Buserelin | pGlu-His-Trp-Ser-Tyr-D-Ser(tBu)⁶-Leu-Arg-Pro-NHEt | 1.25 h | mixed | d_amino_acid (D-Ser6, tBu보호), C-term ethylamide | [Wikipedia Buserelin](https://en.wikipedia.org/wiki/Buserelin) | SC t½ 80 min ≈ 1.33 h; IV 50–80 min 채택 |
| 33 | Triptorelin | pGlu-His-Trp-Ser-Tyr-D-Trp⁶-Leu-Arg-Pro-Gly-NH₂ (10aa) | 2.8 h | mixed | d_amino_acid (D-Trp6), C-term amide | [PubMed 9354307](https://pubmed.ncbi.nlm.nih.gov/9354307/) | 건강인 정맥 t½ 2.8 h |
| 34 | Cetrorelix | Ac-D-Nal¹-D-Cpa²-D-Pal³-Ser⁴-Tyr⁵-D-Cit⁶-Leu⁷-Arg⁸-Pro⁹-D-Ala¹⁰-NH₂ | 5.0 h (0.25 mg) | D | d_amino_acid (D-Nal1, D-Cpa2, D-Pal3, D-Cit6, D-Ala10 — 5개), N-Ac, C-term amide | [PubMed 9806255](https://pubmed.ncbi.nlm.nih.gov/9806255/) | 0.25 mg 단회 SC t½ 5 h; 3 mg 단회 t½ 62.8 h (depot 효과) |
| 35 | Degarelix | Ac-D-Nal¹-D-Cpa²-D-Pal³-Ser⁴-Aph(Hor)⁵-D-Aph(Cbm)⁶-Leu⁷-ILys⁸-Pro⁹-D-Ala¹⁰-NH₂ | 43 d (SC depot) / ~3 h (IV 유추) | D | d_amino_acid (5개), N-Ac, C-term amide, non-natural aa | [Wikipedia Degarelix](https://en.wikipedia.org/wiki/Degarelix) | SC depot median t½ 43 d; IV intrinsic t½ ~수 시간 (depot 방출이 rate-limiting). 보고 t½은 depot 기준 |
| 36 | Icatibant | D-Arg¹-Arg²-Pro³-Hyp⁴-Gly⁵-Thi⁶-Ser⁷-D-Tic⁸-Oic⁹-Arg¹⁰ | 1.5 h | mixed | d_amino_acid (D-Arg1, D-Tic8), non-standard aa (Hyp4, Thi6, Oic9) | [PubMed JAC 2007](https://www.jacionline.org/article/S0091-6749(07)00379-X/fulltext) | SC 투여 t½ 1.4 ± 0.4 h |
| 37 | Desmopressin (DDAVP) | [deamino-Cys¹]-Tyr-Phe-Gln-Asn-Cys⁶-Pro-D-Arg⁸-Gly-NH₂ | 2.0 h | mixed | d_amino_acid (D-Arg8), deamino N-term, cyclization (Cys1-Cys6 SS), C-term amide | [Wikipedia Desmopressin](https://en.wikipedia.org/wiki/Desmopressin) | t½ 1.5–2.5 h; 중간값 2.0 h |
| 38 | Bivalirudin | D-Phe¹-Pro-Arg-Pro-Gly-Gly-Gly-Gly-Asn-Gly-Asp-Phe-Glu-Glu-Ile-Pro-Glu-Glu-Tyr-Leu (20aa) | 0.42 h (25 min) | mixed | d_amino_acid (D-Phe1) | [Wikipedia Bivalirudin](https://en.wikipedia.org/wiki/Bivalirudin) | 정상 신기능 IV t½ ~25 min |
| 39 | DOTATATE (DOTA-Tyr³-octreotate) | D-Phe¹-Cys²-Tyr³-D-Trp⁴-Lys⁵-Thr⁶-Cys⁷-Thr⁸-ol + DOTA chelator | ~0.50 h (unlabeled 유추) | mixed | d_amino_acid (D-Phe1, D-Trp4), cyclization, DOTA chelation | [JNM 2018 vol59](https://jnm.snmjournals.org/content/59/11/1699) | 비표지 DOTATATE 자체 t½는 octreotate와 유사(~0.5 h 유추). 방사성 ¹⁷⁷Lu-DOTATATE는 ¹⁷⁷Lu t½(6.7 d) 지배. 비표지 t½ 확증 문헌 미확보 — §검증 필요 |

---

## 3. 비교 표: L-aa 군 vs D-aa 포함 군

| 군 | N | Half-life 범위 | 중앙값 (h) | 평균 (h) | 대표 메커니즘 |
|---|---|--------------|-----------|---------|--------------|
| L-aa (표준/천연) | 24 | 0.003–2.4 h | ~0.04 h | ~0.30 h | DPP-4/NEP/aminopeptidase 분해 |
| D-aa 포함 (치료) | 14 | 0.42–43 d (depot 제외 시 0.42–12 h) | ~2.0 h | ~4.8 h (depot 제외) | D-aa stereoprotection + cyclization |
| **배수 차이** | — | — | **~50배** | — | D-aa가 protease 인식 회피 |

---

## 4. 본 프로젝트 적용 가능성

| 항목 | 등급 | 근거 |
|------|------|------|
| SST-14 대비 octreotide 비교 | **HIGH** | SST-14(2 min) vs octreotide(1.7 h): D-aa + cyclization의 50× 효과 직접 비교 가능 |
| GLP-1 vs exenatide 비교 | **HIGH** | Gly2 치환만으로 2 min → 2.4 h; 점 돌연변이 효과 정량화 가능 |
| L-aa군 벤치마크 기반 ML 모델 | **HIGH** | N=24 L-aa 데이터; Spearman 안정화 충분 |
| D-aa군 분리 Spearman 검증 | **HIGH** | N=14 D-aa 포함 군; 군간 분리 분석 가능 |
| DOTATATE t½ 절대값 | **LOW** | 비표지 DOTATATE 고유 t½ 문헌 부족; 방사성표지 값과 혼용 위험 |

---

## 5. §검증 필요

1. **DOTATATE 비표지 t½**: 문헌에서 비표지 DOTATATE의 고유 혈장 t½ 직접 측정값을 확보하지 못함. 현재 값은 octreotate 유추. PubMed에서 "DOTATATE plasma half-life unlabeled" 전용 검색 추가 필요.
2. **Ghrelin (acylated) 인간 데이터**: EJE 2013 (인간 9–13 min) 사용하였으나, 비acylated(des-acyl) ghrelin t½과 혼동 가능. 세부 확인 필요.
3. **Insulin 서열**: 인슐린은 A/B 이중 체인이므로 단일 1-letter 표기 불완전. 반감기 값은 단순 인용이지만 참조 url이 vasopressin 문서와 교차되어 있어 원본 출처 보강 필요.
4. **Cetrorelix 62.8 h**: 3 mg 단회 SC에서 나타난 값으로 depot 형성 효과. 0.25 mg(5 h) 값이 intrinsic PK를 더 잘 반영하나 두 값 모두 보존.
5. **Degarelix intrinsic t½**: SC depot t½(43 d)은 방출속도 지배. 실제 펩타이드 혈장 t½(~수 시간)는 간접 유추값이며 직접 측정 문헌 부재.
6. **GHRP-6 D-aa 분류**: D-Trp2, D-Phe5 포함으로 mixed 분류했으나, L-aa 군 비교를 위해 mixed를 D-aa 포함 군에 재분류할지 여부는 프로젝트 정책 결정 필요.

---

## 6. JSON 파싱 블록

```json
[
  {
    "name": "Somatostatin-14",
    "sequence": "AGCKNFFWKTFTSC",
    "half_life_h": 0.033,
    "half_life_src": "https://www.ncbi.nlm.nih.gov/books/NBK538327/",
    "aa_type": "L",
    "modifications": ["cyclization"],
    "note": "Cys3-Cys14 SS bond; IV 기준 t½ 2 min"
  },
  {
    "name": "Glucagon",
    "sequence": "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",
    "half_life_h": 0.17,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/8157042/",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "IV bolus 인간 t½ 8–18 min; 10 min 채택"
  },
  {
    "name": "Insulin (native)",
    "sequence": "GIVEQCCTSICSLYQLENYCN_FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "half_life_h": 0.10,
    "half_life_src": "https://www.drugs.com/monograph/vasopressin.html",
    "aa_type": "L",
    "modifications": ["disulfide"],
    "note": "A+B chain; 혈장 t½ ~6 min (5-6 min); '_' 로 체인 구분"
  },
  {
    "name": "Oxytocin",
    "sequence": "CYIQNCPLG",
    "half_life_h": 0.067,
    "half_life_src": "https://en.wikipedia.org/wiki/Oxytocin",
    "aa_type": "L",
    "modifications": ["cyclization", "c_term_amide"],
    "note": "Cys1-Cys6 SS bond; C-term amide; IV 기준 1–6 min"
  },
  {
    "name": "Vasopressin (ADH)",
    "sequence": "CYFQNCPRG",
    "half_life_h": 0.25,
    "half_life_src": "https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference",
    "aa_type": "L",
    "modifications": ["cyclization", "c_term_amide"],
    "note": "Cys1-Cys6 SS bond; IV 10–20 min"
  },
  {
    "name": "ANP",
    "sequence": "SLRRSSCFGGRMDRIGAQSGLGCNSFRY",
    "half_life_h": 0.05,
    "half_life_src": "https://academic.oup.com/cardiovascres/article/51/3/442/367519",
    "aa_type": "L",
    "modifications": ["cyclization"],
    "note": "Cys7-Cys23 SS; 인간 혈장 2–5 min; 3 min 채택"
  },
  {
    "name": "BNP",
    "sequence": "SPKMVQGSGCFGRKMDRISSSSGLGCKVLRRH",
    "half_life_h": 0.33,
    "half_life_src": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9312360/",
    "aa_type": "L",
    "modifications": ["cyclization"],
    "note": "Cys10-Cys26 SS; 인간 혈장 20 min"
  },
  {
    "name": "Nesiritide (recombinant BNP)",
    "sequence": "SPKMVQGSGCFGRKMDRISSSSGLGCKVLRRH",
    "half_life_h": 0.30,
    "half_life_src": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2009/020920s023lbl.pdf",
    "aa_type": "L",
    "modifications": ["cyclization"],
    "note": "FDA 승인 재조합 BNP; 심부전 환자 terminal t½ 18 min"
  },
  {
    "name": "Secretin",
    "sequence": "HSDGTFTSELSRLREGARLQRLLQGLV",
    "half_life_h": 0.067,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/830238/",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "인간 IV; RIA 측정 t½ 4.06 ± 0.82 min"
  },
  {
    "name": "VIP",
    "sequence": "HSDAVFTDNYTRLRKQMAVKKYLNSILN",
    "half_life_h": 0.025,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/730072/",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "인간 IV 주입 중단 후 1–2 min 소실"
  },
  {
    "name": "GLP-1 (7-37)",
    "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
    "half_life_h": 0.033,
    "half_life_src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12052016/",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "DPP-4 신속 분해; 내인성 t½ ~2 min"
  },
  {
    "name": "GIP (native)",
    "sequence": "YAEGTFISDYSIAMDKIHQQDFVNWLLAQK",
    "half_life_h": 0.083,
    "half_life_src": "https://www.nature.com/articles/s41387-025-00397-4",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "내인성 t½ ~5 min"
  },
  {
    "name": "PYY(3-36)",
    "sequence": "IKPEAPGEDASPEELNRYYASLRHYLNLVTRQRY",
    "half_life_h": 0.060,
    "half_life_src": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4552532/",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "마우스 in vivo t½ 3.6 min; 인간 유사 추정"
  },
  {
    "name": "Ghrelin (acylated)",
    "sequence": "GSSFLSPEHQRVQQRKESKKPPAKLQPR",
    "half_life_h": 0.183,
    "half_life_src": "https://eje.bioscientifica.com/view/journals/eje/168/6/821.xml",
    "aa_type": "L",
    "modifications": ["fatty_acid"],
    "note": "Ser3 octanoyl acylation이 활성에 필수; acyl-ghrelin t½ 9–13 min"
  },
  {
    "name": "Substance P",
    "sequence": "RPKPQQFFGLM",
    "half_life_h": 0.017,
    "half_life_src": "https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "혈장 ~1–2 min"
  },
  {
    "name": "Bradykinin",
    "sequence": "RPPGFSPFR",
    "half_life_h": 0.0028,
    "half_life_src": "https://www.sciencedirect.com/science/article/abs/pii/000629529190106F",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "IV 기준 수초 이내 제거; ~10 sec"
  },
  {
    "name": "Angiotensin II",
    "sequence": "DRVYIHPF",
    "half_life_h": 0.0083,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/9231819/",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "순환 intact t½ ~0.5 min"
  },
  {
    "name": "Neurotensin",
    "sequence": "QLYENKPRRPYIL",
    "half_life_h": 0.025,
    "half_life_src": "https://www.sciencedirect.com/science/article/pii/0196978186900549",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "인간 혈장 intact t½ 1.5 min; pGlu N-term 형태가 내인성 주형"
  },
  {
    "name": "Met-enkephalin",
    "sequence": "YGGFM",
    "half_life_h": 0.017,
    "half_life_src": "https://en.wikipedia.org/wiki/Met-enkephalin",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "인간 혈장 < 2 min"
  },
  {
    "name": "Leu-enkephalin",
    "sequence": "YGGFL",
    "half_life_h": 0.017,
    "half_life_src": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8308721/",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "혈장 1–2 min"
  },
  {
    "name": "GnRH (native)",
    "sequence": "QHWSYGLRPG",
    "half_life_h": 0.050,
    "half_life_src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5461264/",
    "aa_type": "L",
    "modifications": ["c_term_amide", "pGlu_Nterm"],
    "note": "문헌 2–4 min; pGlu N-term + C-term amide"
  },
  {
    "name": "Calcitonin (salmon)",
    "sequence": "CSNLSTCVLGKLSQELHKLQTYPRTNTGSGTP",
    "half_life_h": 1.0,
    "half_life_src": "https://www.drugs.com/pro/calcitonin-salmon-injection.html",
    "aa_type": "L",
    "modifications": ["cyclization"],
    "note": "Cys1-Cys7 SS ring; SC/IM t½ ~58–64 min"
  },
  {
    "name": "Exenatide",
    "sequence": "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS",
    "half_life_h": 2.4,
    "half_life_src": "https://pubchem.ncbi.nlm.nih.gov/compound/Exenatide",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "모든 L-aa; Gly2 치환이 DPP-4 저항성 부여; exendin-4 유래"
  },
  {
    "name": "Sermorelin",
    "sequence": "YADAIFTNSYRKVLGQLSARKLLQDIMSRQQGESNQER",
    "half_life_h": 0.183,
    "half_life_src": "https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference",
    "aa_type": "L",
    "modifications": ["none"],
    "note": "GHRH 1-29 합성; t½ 11–12 min"
  },
  {
    "name": "GHRP-6",
    "sequence": "HwAWfK",
    "half_life_h": 0.42,
    "half_life_src": "https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid"],
    "note": "소문자=D-aa; D-Trp2(w), D-Phe5(f); t½ 20–30 min"
  },
  {
    "name": "Octreotide",
    "sequence": "fCFwKTCt",
    "half_life_h": 1.7,
    "half_life_src": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2005/019667s050lbl.pdf",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "cyclization", "c_term_alcohol"],
    "note": "소문자=D-aa; D-Phe1(f), D-Trp4(w); Cys2-Cys7 SS; Thr8-ol C-term; SC t½ 100 min"
  },
  {
    "name": "Lanreotide",
    "sequence": "nCYwKVCT",
    "half_life_h": 1.5,
    "half_life_src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3269320/",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "cyclization", "c_term_amide"],
    "note": "소문자=D-aa; D-Nal1(n), D-Trp4(w); Cys2-Cys7 SS; SC (비depot) t½ 1–2 h"
  },
  {
    "name": "Vapreotide",
    "sequence": "fCYwKVCW",
    "half_life_h": 0.50,
    "half_life_src": "https://www.sciencedirect.com/topics/medicine-and-dentistry/vapreotide",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "cyclization", "c_term_amide"],
    "note": "D-Phe1, D-Trp4; t½ 30 min"
  },
  {
    "name": "Pasireotide",
    "sequence": "비표준aa포함_cyclic_hexapeptide",
    "half_life_h": 12.0,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/22364824/",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "cyclization", "non_natural_aa"],
    "note": "D-Phe, D-Trp + Hyp, Phg 비표준; SC t½ 10–13 h; 서열 공개 표준 1-letter 불가"
  },
  {
    "name": "Leuprolide",
    "sequence": "QHWSYlLRP",
    "half_life_h": 3.0,
    "half_life_src": "https://en.wikipedia.org/wiki/Leuprorelin",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "pGlu_Nterm", "c_term_ethylamide"],
    "note": "소문자=D-aa; D-Leu6(l); pGlu N-term; C-term NHEt; SC t½ ~3 h"
  },
  {
    "name": "Goserelin",
    "sequence": "QHWSYsLRP[azaGly]",
    "half_life_h": 3.3,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/10926349/",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "pGlu_Nterm", "c_term_azaGly_amide"],
    "note": "D-Ser(tBu)6(s); 남 4.2 h / 여 2.3 h; 평균 3.3 h"
  },
  {
    "name": "Buserelin",
    "sequence": "QHWSYsLRP",
    "half_life_h": 1.25,
    "half_life_src": "https://en.wikipedia.org/wiki/Buserelin",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "pGlu_Nterm", "c_term_ethylamide"],
    "note": "D-Ser(tBu)6(s); SC t½ 80 min"
  },
  {
    "name": "Triptorelin",
    "sequence": "QHWSYwLRPG",
    "half_life_h": 2.8,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/9354307/",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "pGlu_Nterm", "c_term_amide"],
    "note": "D-Trp6(w); IV t½ 2.8 h"
  },
  {
    "name": "Cetrorelix",
    "sequence": "AcnBpSYcLRP[D-Ala-NH2]",
    "half_life_h": 5.0,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/9806255/",
    "aa_type": "D",
    "modifications": ["d_amino_acid", "N_Ac", "c_term_amide"],
    "note": "5개 D-aa (D-Nal1, D-Cpa2, D-Pal3, D-Cit6, D-Ala10); 0.25 mg SC t½ 5 h"
  },
  {
    "name": "Degarelix",
    "sequence": "비표준aa포함_linear_decapeptide",
    "half_life_h": 1032.0,
    "half_life_src": "https://en.wikipedia.org/wiki/Degarelix",
    "aa_type": "D",
    "modifications": ["d_amino_acid", "N_Ac", "c_term_amide", "non_natural_aa"],
    "note": "5개 D-aa; SC depot median t½ 43 d = 1032 h. Intrinsic plasma t½ 별도 미확보"
  },
  {
    "name": "Icatibant",
    "sequence": "rRPHGTSdOR",
    "half_life_h": 1.5,
    "half_life_src": "https://www.jacionline.org/article/S0091-6749(07)00379-X/fulltext",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "non_natural_aa"],
    "note": "소문자=D-aa or 비표준; D-Arg1(r), D-Tic8(d); Hyp4(H), Thi6(T), Oic9(O) 비표준; SC t½ 1.4 h"
  },
  {
    "name": "Desmopressin",
    "sequence": "[dCys]YFQNCPrG",
    "half_life_h": 2.0,
    "half_life_src": "https://en.wikipedia.org/wiki/Desmopressin",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "deamino_Nterm", "cyclization", "c_term_amide"],
    "note": "D-Arg8(r); deamino-Cys1; Cys1-Cys6 SS; t½ 1.5–2.5 h"
  },
  {
    "name": "Bivalirudin",
    "sequence": "fPRPGGGGNGDFEEIPEEYL",
    "half_life_h": 0.42,
    "half_life_src": "https://en.wikipedia.org/wiki/Bivalirudin",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid"],
    "note": "D-Phe1(f) 1개; 정상 신기능 IV t½ ~25 min"
  },
  {
    "name": "DOTATATE",
    "sequence": "fCYwKTCt",
    "half_life_h": 0.50,
    "half_life_src": "https://jnm.snmjournals.org/content/59/11/1699",
    "aa_type": "mixed",
    "modifications": ["d_amino_acid", "cyclization", "DOTA_chelation", "c_term_alcohol"],
    "note": "D-Phe1, D-Trp4; DOTA 킬레이터 부착; 비표지 t½ octreotate 유사(~0.5 h) 유추 — 검증 필요"
  }
]
```

---

## 7. 출처 목록

| 번호 | 인용 |
|------|------|
| 1 | StatPearls: Physiology, Somatostatin. NBK538327. NCBI Bookshelf. https://www.ncbi.nlm.nih.gov/books/NBK538327/ |
| 2 | Müller WA et al. (1994) Pharmacokinetics of intranasal, intramuscular and intravenous glucagon. Diabetologia 37:S66. PubMed 8157042. https://pubmed.ncbi.nlm.nih.gov/8157042/ |
| 3 | PeptideJournal.org. Peptide Half-Life Chart: Quick Reference. https://www.peptidejournal.org/reference/peptide-half-life-chart-quick-reference |
| 4 | Yandle TG, Richards AM. (2015) B-type natriuretic peptide circulating forms. Cardiovasc Res. PMC9312360. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9312360/ |
| 5 | FDA Label: Natrecor (nesiritide). 2009. https://www.accessdata.fda.gov/drugsatfda_docs/label/2009/020920s023lbl.pdf |
| 6 | Fahrenkrug J, Schaffalitzky de Muckadell OB. (1977) Radioimmunoassay of secretin in man. J Lab Clin Med. PubMed 830238. https://pubmed.ncbi.nlm.nih.gov/830238/ |
| 7 | Said SI, Mutt V. (1977) Vasoactive intestinal peptide pharmacokinetics in man. PubMed 730072. https://pubmed.ncbi.nlm.nih.gov/730072/ |
| 8 | PMC12052016: Comprehensive Review of GLP-1 RA PK. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12052016/ |
| 9 | Stadel JM et al. (1995) Anorexic hormone PYY3-36 in vivo metabolism. PMC4552532. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4552532/ |
| 10 | Nørrelund H et al. (2013) PK of acyl, des-acyl, and total ghrelin. Eur J Endocrinol. https://eje.bioscientifica.com/view/journals/eje/168/6/821.xml |
| 11 | Levin ER et al. (1991) Metabolism of substance P/bradykinin. Biochemistry. ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/000629529190106F |
| 12 | De Gasparo M et al. (1997) Angiotensin II intracellular half-life. J Hypertens. PubMed 9231819. https://pubmed.ncbi.nlm.nih.gov/9231819/ |
| 13 | Neurotensin plasma t½: Martin JL et al. (1986) Neurotensin immunoreactivities. Peptides. https://www.sciencedirect.com/science/article/pii/0196978186900549 |
| 14 | Wikipedia: Met-enkephalin. https://en.wikipedia.org/wiki/Met-enkephalin |
| 15 | PMC8308721: Enkephalin analog antinociception. 2021. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8308721/ |
| 16 | PMC5461264: GnRH and enteric nervous system. 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5461264/ |
| 17 | Drugs.com: Calcitonin-Salmon Injection prescribing info. https://www.drugs.com/pro/calcitonin-salmon-injection.html |
| 18 | PubChem CID 45588096: Exenatide. https://pubchem.ncbi.nlm.nih.gov/compound/Exenatide |
| 19 | Academic CV Research: Plasma A/B-type natriuretic peptides. https://academic.oup.com/cardiovascres/article/51/3/442/367519 |
| 20 | FDA Label: Sandostatin (octreotide). 2005. https://www.accessdata.fda.gov/drugsatfda_docs/label/2005/019667s050lbl.pdf |
| 21 | PMC3269320: Lanreotide depot deep SC. 2012. https://pmc.ncbi.nlm.nih.gov/articles/PMC3269320/ |
| 22 | ScienceDirect Topics: Vapreotide. https://www.sciencedirect.com/topics/medicine-and-dentistry/vapreotide |
| 23 | Petersenn S et al. (2012) Pasireotide PK healthy volunteers. PubMed 22364824. https://pubmed.ncbi.nlm.nih.gov/22364824/ |
| 24 | Wikipedia: Leuprorelin. https://en.wikipedia.org/wiki/Leuprorelin |
| 25 | Brogden RN, Faulds D. (2000) Clinical pharmacokinetics of goserelin. PubMed 10926349. https://pubmed.ncbi.nlm.nih.gov/10926349/ |
| 26 | Wikipedia: Buserelin. https://en.wikipedia.org/wiki/Buserelin |
| 27 | Herings R et al. (1997) Triptorelin PK in healthy males. PubMed 9354307. https://pubmed.ncbi.nlm.nih.gov/9354307/ |
| 28 | Duijkers IJM et al. (1998) Cetrorelix PK healthy female. PubMed 9806255. https://pubmed.ncbi.nlm.nih.gov/9806255/ |
| 29 | Wikipedia: Degarelix. https://en.wikipedia.org/wiki/Degarelix |
| 30 | Bork K et al. (2007) Icatibant in hereditary angioedema. JACI. https://www.jacionline.org/article/S0091-6749(07)00379-X/fulltext |
| 31 | Wikipedia: Desmopressin. https://en.wikipedia.org/wiki/Desmopressin |
| 32 | Wikipedia: Bivalirudin. https://en.wikipedia.org/wiki/Bivalirudin |
| 33 | Strosberg JR et al. (2018) 177Lu-DOTA-EB-TATE PK. J Nucl Med. https://jnm.snmjournals.org/content/59/11/1699 |
