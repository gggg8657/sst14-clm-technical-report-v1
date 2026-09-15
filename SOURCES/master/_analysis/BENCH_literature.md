# 펩타이드 벤치마크 Ground-Truth 데이터셋

**목적**: 문헌에 공개된 펩타이드들의 실측값과 우리 파이프라인 예측 순위 비교  
**수집일**: 2026-06-17  
**수집 담당**: researcher (Claude Sonnet 4.6)

---

## 1. 마크다운 요약 표

### 범례
- **t½**: 혈청/혈장 반감기 (정맥주사 기준 우선, 없으면 피하주사)
- **HC50/독성**: 용혈 활성 HC50 또는 관련 독성 지표
- **변형**: D-아미노산, 아실화, PEG화, 환형화(비-SS bond) 등 비표준 변형 유무
- **[불확실]**: 서열 또는 수치 확인이 불완전한 항목

| # | name | sequence (1-letter) | t½ (값+단위) | t½ 출처 | HC50/독성 | 독성 출처 | 비고(변형) |
|---|------|---------------------|-------------|---------|-----------|-----------|-----------|
| 1 | Somatostatin-14 (SST-14) | AGCKNFFWKTFTSC | <3 min | Wikipedia Somatostatin; [PMC6152110](https://pmc.ncbi.nlm.nih.gov/articles/PMC6152110/) | 보고 없음 | — | Cys3-Cys14 SS bond (환형); 천연 펩타이드 |
| 2 | Octreotide | DfCFwKTCT-ol [주1] | ~100 min (SC); IV 이중지수: t½α 10 min, t½β 90 min | Wikipedia Octreotide | 보고 없음 (저독성) | — | D-Phe1, D-Trp4 치환; C-말단 threoninol; Cys2-Cys7 SS bond; **모델 직접 입력 불가** (D-aa, 비표준 C말단) |
| 3 | Octreotate (TATE) | DfCYwKTCT-OH [주2] | 약 100 min (octreotide와 동등 추정) | [PMC6152110](https://pmc.ncbi.nlm.nih.gov/articles/PMC6152110/) (추론) | 보고 없음 | — | D-Phe1, Tyr3 치환, D-Trp4; Cys SS bond; **모델 직접 입력 불가** |
| 4 | DOTATATE (177Lu-DOTATATE) | DfCYwKTCT + DOTA chelator [주3] | 생물학적 t½ 혈중 3.5 ±1.4 h (유효), 71 ±28 h (말단) | [PMC9898489](https://pmc.ncbi.nlm.nih.gov/articles/PMC9898489/) | 저독성 (임상 사용) | — | D-Phe1, Tyr3, D-Trp4; DOTA 킬레이터 N-말단 결합; 방사성핵종 라벨; **모델 직접 입력 불가** |
| 5 | Lanreotide | (d-2-Nal)CY(d-W)KVCT-NH2 [주4] | IR: ~2 h; SR depot: ~5 days | Wikipedia Lanreotide | 보고 없음 | — | D-2-Nal1, D-Trp4; C-말단 아미드; Cys SS bond; **모델 직접 입력 불가** |
| 6 | Pasireotide | 환형 헥사펩타이드 [Phg-D-Trp-Lys-Tyr(Bzl)-Phe-Pro*] [주5] | 7–12 h (SC) | [PMC9156514](https://pmc.ncbi.nlm.nih.gov/articles/PMC9156514/) | 저독성 (임상 사용) | — | D-Trp; Phenylglycine(비표준); modified Pro; 환형(비-SS); **모델 직접 입력 불가** |
| 7 | GLP-1(7-37) (native) | HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG | ~1.5–2 min | [PMC5401818](https://pmc.ncbi.nlm.nih.gov/articles/PMC5401818/) | 보고 없음 | — | 표준 선형; DPP-4에 의해 빠른 분해; 모델 입력 적합 |
| 8 | Liraglutide | HAEGTFTSDVSSYLEGQAAKEFIAWLVRGGG [주6] | ~13 h (SC) | [ResearchGate fig](https://www.researchgate.net/figure/Structure-and-half-lives-of-native-human-GLP-1-liraglutide-and-semaglutide_fig1_337691592) | 보고 없음 | — | Lys26에 C16 지방산(palmitoyl)-γGlu 결합; Lys34→Arg 치환; 모델 입력 시 지방산 변형 메모 필요 |
| 9 | Semaglutide | H-Aib-EGTFTSDVSSYLEGQAAKEFIAWLVRGR [주7] | ~165 h (7 days) (SC) | [PNAS doi:10.1073/pnas.2415815121](https://www.pnas.org/doi/10.1073/pnas.2415815121) | 보고 없음 | — | Ala8→Aib(비표준); Lys26-C18 이산지방산 결합; Lys34→Arg; **Aib 비표준, 모델 주의** |
| 10 | Exenatide (Exendin-4) | HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS | 2.4 h (SC, mean terminal) | [PubMed 25723538](https://pubmed.ncbi.nlm.nih.gov/25723538/) | 보고 없음 (저독성) | — | 표준 39aa 선형; Gila monster 유래; DPP-4 저항성; 모델 입력 적합 |
| 11 | Human Insulin | A체인: GIVEQCCTSICSLYQLENYCN; B체인: FVNQHLCGSHLVEALYLVCGERGFFYTPKT | ~4–5 min (IV, 내인성); 피하 제형 다양 | [ScienceDirect S0026049568900097](https://www.sciencedirect.com/science/article/abs/pii/0026049568900097) | 과량 시 저혈당 (독성 지표 아님) | — | A+B 이중체인; 3개 SS bond; 모델에 단일 사슬로 입력 불가 (비표준 구조) |
| 12 | Oxytocin | CYIQNCPLG [주8] | 1–6 min (IV); 3–5 min (일반인용) | FDA Pitocin label (accessdata.fda.gov/drugsatfda_docs/label/2014/018261s031lbl.pdf) | 보고 없음 (호르몬 투여 용량에서 안전) | — | Cys1-Cys6 SS bond; C-말단 글리신아미드(-NH2); 환형; 모델 입력 시 변형 메모 |
| 13 | Vasopressin (AVP) | CYFQNCPRG [주9] | 10–20 min (IV) | Wikipedia Vasopressin; [medicine.com](https://www.medicine.com/drug/vasopressin/hcp) | 보고 없음 | — | Cys1-Cys6 SS bond; C-말단 글리신아미드; 환형; 모델 입력 시 변형 메모 |
| 14 | Bradykinin | RPPGFSPFR | 15–30 sec (in vivo); ~17 sec | [AJP-Heart doi:10.1152/ajpheart.2001.280.5.H2182](https://journals.physiology.org/doi/full/10.1152/ajpheart.2001.280.5.H2182) | 보고 없음 | — | 표준 9aa 선형; ACE/kininase II에 의해 빠른 분해; 모델 입력 적합 |
| 15 | Substance P | RPKPQQFFGLM | 수십 초–수분 (조직); 혈장에서는 더 길다(수 시간 보고도 있음) [불확실] | [PubMed NBK554583](https://www.ncbi.nlm.nih.gov/books/NBK554583/) | 보고 없음 | — | 표준 11aa 선형; NEP/ACE에 의해 분해; 모델 입력 적합 |
| 16 | Melittin | GIGAVLKVLTTGLPALISWIKRKRQQ | ~24 min (IV, in vivo) | [PMC4740973](https://pmc.ncbi.nlm.nih.gov/articles/PMC4740973/) | HC50 16.28 ±0.17 µg/mL (인간 RBC 2% suspension); HD50 1.9 ±0.1 µM | [PMC6208649](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6208649/); [PMC4740973](https://pmc.ncbi.nlm.nih.gov/articles/PMC4740973/) | 표준 26aa 선형; 강한 양이온성/양친매성; 모델 입력 적합; 독성 양성대조군으로 유용 |
| 17 | Bivalirudin | FPRPGGGGNGDFEEIPEEYL [주10] | 25 min (정상 신기능, IV) | Wikipedia Bivalirudin; [PMC5760933](https://pmc.ncbi.nlm.nih.gov/articles/PMC5760933/) | 보고 없음 (저독성 항응고제) | — | D-Phe1 치환; 단순 선형(SS bond 없음); **D-aa, 모델 주의** |

---

## 2. 주석

- **[주1]** 옥트레오타이드 표준 서열: D-Phe-Cys-Phe-D-Trp-Lys-Thr-Cys-Thr-ol. 1-letter 혼합 표기(소문자=D-aa)가 관행이나 표준 1-letter 코드 아님. 표준 유사서열: `FCFWKTCT` (D-aa 무시 시); 변형 메모 필수.
- **[주2]** Octreotate = octreotide의 C-말단 threoninol → threonine-OH 변환. 서열: D-Phe-Cys-Tyr-D-Trp-Lys-Thr-Cys-Thr-OH. 1-letter 유사: `FCYWKTCT`.
- **[주3]** DOTATATE = DOTA-[Tyr3]-octreotate. 펩타이드 서열은 Octreotate와 동일; DOTA는 N-말단에 화학결합. 1-letter 서열: `FCYWKTCT` (DOTA 제외).
- **[주4]** 란레오타이드 서열: D-2-Nal-Cys-Tyr-D-Trp-Lys-Val-Cys-Thr-NH2. 2-Nal(2-naphthylalanine)은 비표준. 표준 유사: `FCYWKVCT` (D-aa, Nal 무시 시).
- **[주5]** 파시레오타이드: 환형 헥사펩타이드. 구성: Phg(phenylglycine)-D-Trp-Lys-Tyr(OBn)-Phe-hydroxyPro. 표준 1-letter 근사: `FWKYFP` (비표준 잔기 근사). **매우 불완전한 근사**—실제 서열 검증 필요.
- **[주6]** 리라글루타이드 기본 펩타이드 서열: GLP-1(7-37) 대비 Lys34→Arg, Lys26에 C16 지방산 부착. 표준 서열 부분(아실화 제외): `HAEGTFTSDVSSYLEGQAAKEFIAWLVRG`. 피하 경로 t½.
- **[주7]** 세마글루타이드: Ala8→Aib(2-aminoisobutyric acid, 비표준), Lys26에 C18 이산지방산, Lys34→Arg. 표준 근사 서열(Aib→A): `HAEGTFTSDVSSYLEGQAAKEFIAWLVRGR`. 피하 경로 t½.
- **[주8]** 옥시토신: 9aa 환형펩타이드, C-말단 아미드(-NH2). Cys1-Cys6 SS bond. 서열 확인: Wikipedia 'Oxytocin' + EBI CHEBI:7872.
- **[주9]** 바소프레신(AVP): 9aa 환형. Cys1-Cys6 SS bond, C-말단 아미드. Arg8 위치가 oxytocin Leu8과 차이. 서열: Wikipedia 'Vasopressin'.
- **[주10]** 비발리루딘: D-Phe1. 나머지 19aa는 표준 L-아미노산. 표준 유사 서열(D-Phe 무시): `FPRPGGGGNGDFEEIPEEYL`.

---

## 3. t½ 단위 환산 요약 (시간 기준)

| name | t½ 원문 | t½ 환산(h) | 환산 가정 |
|------|---------|-----------|---------|
| SST-14 | <3 min | <0.05 h | 직접 환산 |
| Octreotide | ~100 min (SC mean) | ~1.67 h | 100/60 |
| GLP-1(7-37) | ~1.5–2 min | ~0.028 h | 중간값 1.75 min |
| Liraglutide | ~13 h | 13.0 h | 직접 |
| Semaglutide | ~165 h | 165.0 h | 직접 |
| Exenatide | 2.4 h | 2.4 h | 직접 |
| Human Insulin | ~4–5 min | ~0.075 h | 중간값 4.5 min |
| Oxytocin | 1–6 min | ~0.05 h | 중간값 3 min |
| Vasopressin | 10–20 min | ~0.25 h | 중간값 15 min |
| Bradykinin | 15–30 sec | ~0.005 h | 중간값 22 sec |
| Substance P | 수분 이내 (불확실) | null | 값 불확실—JSON null |
| Melittin | ~24 min | ~0.4 h | 직접 환산 |
| Bivalirudin | 25 min (정상신기능) | ~0.42 h | 직접 환산 |
| DOTATATE | 3.5 h (유효 혈중) | 3.5 h | 직접 |
| Lanreotide | 2 h (IR) | 2.0 h | IR 기준 |
| Pasireotide | 7–12 h | ~9.5 h | 중간값 |

---

## 4. 검증 필요 사항 (§검증 필요)

1. **Substance P 정확한 t½**: 문헌마다 수십 초(조직)~수 시간(혈장)으로 크게 다름. 방법론(in vivo/ex vivo/species)에 따른 차이 분석 필요. 현재 JSON에 null 처리.
2. **Octreotate/DOTATATE 독립 t½**: 옥트레오테이트 단독 약동학 데이터는 거의 없음. DOTATATE의 t½은 방사성핵종 표지 복합체 기준이라 순수 펩타이드와 다를 수 있음.
3. **Pasireotide 서열**: 비표준 잔기(Phenylglycine, modified Pro) 포함. 우리 모델 1-letter 입력에 적합하지 않음. 처방정보(SOM230 prescribing information) 원문 확인 권장.
4. **Liraglutide/Semaglutide 정확한 펩타이드 서열**: 아실화 제외 기본 서열만 수집; C16/C18 지방산 linker 위치는 우리 모델 한계로 인해 표준 서열로 근사.
5. **Bivalirudin HC50**: 항응고제이므로 용혈 독성 데이터가 드묾. 검색 결과에서 발견 못함.
6. **Oxytocin/Vasopressin C-말단 아미드**: 우리 파이프라인이 C-말단 아미드(-NH2) 처리를 지원하는지 확인 필요—HC50 예측에 영향 가능.
7. **DOTATOC 독립 t½**: DOTATOC(D-Phe1-Tyr3-octreotide + DOTA)의 생물학적 t½ 데이터는 명시적 수치 미발견. 방사성 표지 데이터만 존재.
8. **GLP-1/Semaglutide/Liraglutide HC50**: 이들 항목에 대한 용혈 독성 데이터 없음—우리 파이프라인 hc50 게이트 적용 시 비교 불가.

---

## 5. JSON 블록

```json
[
  {
    "name": "Somatostatin-14",
    "sequence": "AGCKNFFWKTFTSC",
    "half_life_h": 0.05,
    "half_life_src": "https://en.wikipedia.org/wiki/Somatostatin; https://pmc.ncbi.nlm.nih.gov/articles/PMC6152110/",
    "hc50": null,
    "tox_src": null,
    "modified": false,
    "note": "Cys3-Cys14 SS bond 환형; <3 min → 0.05 h 환산(상한값 사용)"
  },
  {
    "name": "Octreotide",
    "sequence": "FCFWKTCT",
    "half_life_h": 1.67,
    "half_life_src": "https://en.wikipedia.org/wiki/Octreotide",
    "hc50": null,
    "tox_src": null,
    "modified": true,
    "note": "D-Phe1, D-Trp4, C-말단 threoninol; 1-letter 서열은 D-aa 및 C-말단 변형 무시한 근사값. ~100 min SC → 1.67 h"
  },
  {
    "name": "Octreotate (TATE)",
    "sequence": "FCYWKTCT",
    "half_life_h": 1.67,
    "half_life_src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6152110/ (octreotide로부터 추론)",
    "hc50": null,
    "tox_src": null,
    "modified": true,
    "note": "D-Phe1, Tyr3, D-Trp4, C-말단-OH; octreotide와 t½ 유사 추정. 1-letter는 D-aa 무시 근사"
  },
  {
    "name": "DOTATATE",
    "sequence": "FCYWKTCT",
    "half_life_h": 3.5,
    "half_life_src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9898489/",
    "hc50": null,
    "tox_src": null,
    "modified": true,
    "note": "N-말단 DOTA 킬레이터 결합, D-Phe1, Tyr3, D-Trp4; 177Lu 표지 복합체 혈중 유효 t½ 3.5±1.4 h. 1-letter는 DOTA 및 D-aa 무시 근사"
  },
  {
    "name": "Lanreotide",
    "sequence": "FCYWKVCT",
    "half_life_h": 2.0,
    "half_life_src": "https://en.wikipedia.org/wiki/Lanreotide",
    "hc50": null,
    "tox_src": null,
    "modified": true,
    "note": "D-2-Nal1, D-Trp4, Val6, C-말단 아미드; IR t½ 2 h (SR depot 5 days 제외). 1-letter는 2-Nal→F, D-aa 무시 근사"
  },
  {
    "name": "Pasireotide",
    "sequence": "FWKYFP",
    "half_life_h": 9.5,
    "half_life_src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9156514/",
    "hc50": null,
    "tox_src": null,
    "modified": true,
    "note": "환형 헥사펩타이드; Phenylglycine(비표준)→F 근사, D-Trp→W, modified Pro→P 근사. 7–12 h 중간값 9.5 h. 서열 매우 불완전한 근사—직접 입력 비권장"
  },
  {
    "name": "GLP-1(7-37)",
    "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG",
    "half_life_h": 0.028,
    "half_life_src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5401818/",
    "hc50": null,
    "tox_src": null,
    "modified": false,
    "note": "표준 선형 31aa; DPP-4에 의해 빠른 분해. 1.75 min 중간값 → 0.028 h"
  },
  {
    "name": "Liraglutide",
    "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVRG",
    "half_life_h": 13.0,
    "half_life_src": "https://www.researchgate.net/figure/Structure-and-half-lives-of-native-human-GLP-1-liraglutide-and-semaglutide_fig1_337691592",
    "hc50": null,
    "tox_src": null,
    "modified": true,
    "note": "Lys34→Arg, Lys26에 C16 palmitoyl-γGlu 아실화; 서열은 지방산 제외 기본 서열 (29aa). 피하주사 SC t½"
  },
  {
    "name": "Semaglutide",
    "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVRGR",
    "half_life_h": 165.0,
    "half_life_src": "https://www.pnas.org/doi/10.1073/pnas.2415815121",
    "hc50": null,
    "tox_src": null,
    "modified": true,
    "note": "Ala8→Aib(비표준, A로 근사), Lys34→Arg, Lys26에 C18 이산지방산; 30aa. 피하주사 SC t½ 165 h(7 days)"
  },
  {
    "name": "Exenatide (Exendin-4)",
    "sequence": "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS",
    "half_life_h": 2.4,
    "half_life_src": "https://pubmed.ncbi.nlm.nih.gov/25723538/",
    "hc50": null,
    "tox_src": null,
    "modified": false,
    "note": "표준 39aa 선형; Gila monster 유래 자연 펩타이드; C-말단 아미드 형태도 존재하나 본 서열은 free acid. SC t½ 2.4 h"
  },
  {
    "name": "Human Insulin",
    "sequence": "GIVEQCCTSICSLYQLENYCN+FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "half_life_h": 0.075,
    "half_life_src": "https://www.sciencedirect.com/article/abs/pii/0026049568900097",
    "hc50": null,
    "tox_src": null,
    "modified": false,
    "note": "A체인(21aa)+B체인(30aa) 이중사슬, 3개 SS bond; 단일 1-letter 입력 불가. 4.5 min 중간값 → 0.075 h. IV 내인성 기준"
  },
  {
    "name": "Oxytocin",
    "sequence": "CYIQNCPLG",
    "half_life_h": 0.05,
    "half_life_src": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/018261s031lbl.pdf",
    "hc50": null,
    "tox_src": null,
    "modified": false,
    "note": "Cys1-Cys6 SS bond 환형; C-말단 glycinamide(-NH2). 1–6 min → 3 min 중간값 → 0.05 h. C-말단 아미드 모델 처리 주의"
  },
  {
    "name": "Vasopressin (AVP)",
    "sequence": "CYFQNCPRG",
    "half_life_h": 0.25,
    "half_life_src": "https://www.medicine.com/drug/vasopressin/hcp; https://en.wikipedia.org/wiki/Vasopressin",
    "hc50": null,
    "tox_src": null,
    "modified": false,
    "note": "Cys1-Cys6 SS bond 환형; C-말단 glycinamide(-NH2). 10–20 min → 15 min 중간값 → 0.25 h. C-말단 아미드 모델 처리 주의"
  },
  {
    "name": "Bradykinin",
    "sequence": "RPPGFSPFR",
    "half_life_h": 0.005,
    "half_life_src": "https://journals.physiology.org/doi/full/10.1152/ajpheart.2001.280.5.H2182",
    "hc50": null,
    "tox_src": null,
    "modified": false,
    "note": "표준 9aa 선형; ACE(kininase II)에 의해 초고속 분해. 15–30 sec → 22 sec 중간값 → 0.006 h (≈0.005 h 반올림)"
  },
  {
    "name": "Substance P",
    "sequence": "RPKPQQFFGLM",
    "half_life_h": null,
    "half_life_src": "https://www.ncbi.nlm.nih.gov/books/NBK554583/",
    "hc50": null,
    "tox_src": null,
    "modified": false,
    "note": "표준 11aa 선형; 조직 내 수십 초, 혈장 내 수 시간으로 문헌마다 크게 다름—신뢰할 수 있는 단일 수치 확보 불가. null 처리"
  },
  {
    "name": "Melittin",
    "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "half_life_h": 0.4,
    "half_life_src": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4740973/",
    "hc50": 16.28,
    "tox_src": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6208649/ (HC50 µg/mL, human RBC 2%); https://pmc.ncbi.nlm.nih.gov/articles/PMC4740973/ (HD50 1.9 µM)",
    "modified": false,
    "note": "표준 26aa 선형 양이온성 양친매성; 독성 양성대조군으로 최적. HC50=16.28 µg/mL, HD50=1.9 µM. ~24 min → 0.4 h"
  },
  {
    "name": "Bivalirudin",
    "sequence": "FPRPGGGGNGDFEEIPEEYL",
    "half_life_h": 0.42,
    "half_life_src": "https://en.wikipedia.org/wiki/Bivalirudin; https://pmc.ncbi.nlm.nih.gov/articles/PMC5760933/",
    "hc50": null,
    "tox_src": null,
    "modified": true,
    "note": "D-Phe1 치환(항응고제); 나머지 19aa 표준 L. 1-letter 서열은 D-Phe 무시 근사. 25 min(정상 신기능, IV) → 0.42 h"
  }
]
```

---

## 6. 참고 문헌 목록

| # | 인용 | URL |
|---|------|-----|
| 1 | Wikipedia – Somatostatin | https://en.wikipedia.org/wiki/Somatostatin |
| 2 | Wikipedia – Octreotide | https://en.wikipedia.org/wiki/Octreotide |
| 3 | Wikipedia – Lanreotide | https://en.wikipedia.org/wiki/Lanreotide |
| 4 | Wikipedia – Bivalirudin | https://en.wikipedia.org/wiki/Bivalirudin |
| 5 | Wikipedia – Vasopressin | https://en.wikipedia.org/wiki/Vasopressin |
| 6 | Roosek et al. 2018, PMC6152110 – Somatostatin analogs in oncology | https://pmc.ncbi.nlm.nih.gov/articles/PMC6152110/ |
| 7 | PMC4740973 – Melittin prodrug nanoparticles; plasma half-life 24 min | https://pmc.ncbi.nlm.nih.gov/articles/PMC4740973/ |
| 8 | PMC6208649 – Melittin HC50 16.28 µg/mL | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6208649/ |
| 9 | PMC5401818 – GLP-1 t½ ~2 min, DPP-4 degradation | https://pmc.ncbi.nlm.nih.gov/articles/PMC5401818/ |
| 10 | PNAS 2024 – Semaglutide t½ 165 h | https://www.pnas.org/doi/10.1073/pnas.2415815121 |
| 11 | ResearchGate – Liraglutide/Semaglutide t½ 비교 | https://www.researchgate.net/figure/Structure-and-half-lives-of-native-human-GLP-1-liraglutide-and-semaglutide_fig1_337691592 |
| 12 | PubMed 25723538 – Exenatide metabolic stability, t½ 2.4 h | https://pubmed.ncbi.nlm.nih.gov/25723538/ |
| 13 | ScienceDirect – Insulin endogenous t½ | https://www.sciencedirect.com/article/abs/pii/0026049568900097 |
| 14 | FDA Pitocin label – Oxytocin t½ 1–6 min | https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/018261s031lbl.pdf |
| 15 | Medicine.com – Vasopressin t½ 10–20 min | https://www.medicine.com/drug/vasopressin/hcp |
| 16 | AJP-Heart 2001 – Bradykinin t½ 15–30 sec, ACE | https://journals.physiology.org/doi/full/10.1152/ajpheart.2001.280.5.H2182 |
| 17 | NCBI NBK554583 – Substance P pharmacology | https://www.ncbi.nlm.nih.gov/books/NBK554583/ |
| 18 | PMC9156514 – Pasireotide t½ 7–12 h | https://pmc.ncbi.nlm.nih.gov/articles/PMC9156514/ |
| 19 | PMC9898489 – DOTATATE PBPK, t½ 3.5 h | https://pmc.ncbi.nlm.nih.gov/articles/PMC9898489/ |
| 20 | PMC5760933 – Bivalirudin LC-MS/MS PK | https://pmc.ncbi.nlm.nih.gov/articles/PMC5760933/ |
