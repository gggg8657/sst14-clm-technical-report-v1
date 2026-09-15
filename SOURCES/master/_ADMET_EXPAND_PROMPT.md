ADMET 용혈 독성 벤치마크 **데이터 확장** 작업이다. 작업 디렉토리: `[LOCAL_PATH]`. 이 디렉토리 밖(특히 `[LOCAL_PATH]`)은 절대 침범 금지.

## 목표
현재 문헌 데이터셋은 26종(L=22, D=4)뿐이고 **D-amino acid 펩타이드가 n=4로 통계적으로 과소**하다. 문헌에서 용혈 활성(hemolytic activity, HC50/MHC) 값이 보고된 펩타이드를 **추가 수집**해 D-aa를 최소 12종 이상, 전체 45종 이상으로 확장한다.

## 절대 규칙 (환각 금지 — 위반 시 작업 무효)
1. **실제 출판 문헌값만.** 모든 신규 항목은 반드시 검증 가능한 `src`(논문 URL/DOI, 가급적 PMC/PubMed/저널)를 가져야 한다.
2. **값을 지어내지 마라.** HC50(µg/mL) 실측값을 출처에서 확인 못 하면 그 항목은 **건너뛴다**. 추정/보간 금지.
3. 출처에서 HC50가 µM 단위면 분자량으로 µg/mL 환산하고 `hc50_uM`에도 원값을 남긴다. 환산 불가하면 `hc50_ugml: null`로 두되 `hemolytic` 라벨(고용혈=true/저용혈=false)은 본문 근거로 채운다.
4. **기존 26종은 그대로 보존.** 신규만 append. 중복 서열 금지.

## 데이터 스키마 (기존과 동일)
각 항목: `{"name","sequence"(1-letter, 표준 20 AA만),"hc50_ugml","hc50_uM","hemolytic"(bool),"aa_type"("L"|"D"|"MIXED"),"note"(측정조건 예: "2% hRBC"),"src"(URL)}`

## 절차
1. 기존 데이터 읽기: `_analysis/BENCH_admet_literature.md` 안의 ```json 블록.
2. WebSearch/WebFetch로 용혈 데이터가 있는 펩타이드 논문을 찾는다. 우선순위: **D-amino acid 항균/세포투과 펩타이드**(D-enantiomer, D-substituted), 다음 L-aa. 검색 예: "D-amino acid peptide hemolytic activity HC50", "all-D antimicrobial peptide hemolysis", "peptide MHC human erythrocyte µg/mL".
3. 각 후보의 서열·HC50·측정조건·출처를 확인해 스키마로 정리. 출처 검증 안 되면 skip.
4. 기존 26 + 신규를 합쳐 **새 파일 `_analysis/BENCH_admet_v2.md`** 에 동일 ```json 포맷으로 쓴다(헤더에 출처·수집일·규칙 명시). aa_type 필드 반드시 채울 것.
5. 벤치마크 실행: `~/miniforge3/envs/bio-tools/bin/python bench_admet_tox_v2.py` (출력 `_analysis/BENCH_admet_v2_results.json` 갱신). 가동 중인 무한엔진과 GPU/파일 충돌 주의 — 실패 시 재시도.
6. 결과 페이지 갱신: `python build.py` 후, `docs/`(GitHub Pages 사본)에도 `_workspace/report_site` → repo `docs/` 동기화가 필요하면 `BENCH_admet_v2_results.json`/page15 변경분 반영.
7. 진행/결과를 `_ADMET_EXPAND_RESULT.md`에 요약(신규 N, L/D 분포, AUC_all/L/D before→after, 추가한 항목 목록+출처).
8. git: 루트 `[LOCAL_PATH]`에서 변경분 commit (한국어 제목+영문 상세+`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`) 후 `git push origin main`. **무한엔진 산출물(runs/)은 add하지 말고** ADMET 확장 관련 파일만 선택 add.

## 제약
- bio-tools conda env로 pepADMET HC50 예측 실행(`predict_toxicity_for_sequences`). .venv 아님.
- fail-closed·실측 우선. 불확실하면 보고하고 skip.
- 다 끝나면 `_ADMET_EXPAND_RESULT.md` 마지막 줄에 `STATUS: DONE` 기록.
