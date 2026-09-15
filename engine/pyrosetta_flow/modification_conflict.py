"""
modification_conflict.py — SST-14 펩타이드 modification 충돌 검사기
====================================================================
step08_stability 수정 없이 독립적으로 동작하는 순수 규칙 기반 검사기.

주요 기능:
  check_modification_conflicts(sequence, mod_list) -> {"ok": bool, "conflicts": list}

충돌 규칙:
  FAIL 규칙 (치명적, 합성 불가 또는 라디오케미컬 실패):
    R-F1: 동일 Lys에 fatty_acid + DOTA 동시 → 단일 ε-amine 충돌
    R-F2: D-aa 치환 위치 + acylation 동일 위치 → WARN으로 완화 (D-Lys NHS-ester 가능, chirality-HPLC 필요)
    R-F3: N-term capping + DOTA N-term 동시 → N-term 점유 충돌
    R-F4: DOTA가 FWKT pharmacophore 구성 Lys(K9) 부착 → 생물활성 소멸 (Reubi 2017: K9→Ala Ki ×50)

  WARN 규칙 (경고, 합성 가능하나 효율 저하 우려):
    R-F2: [완화] D-aa + acylation 동일 위치 → chirality-HPLC 검증 권고
    R-W1: 동일 Lys에 fatty_acid + PEG 동시 → ε-amine 경쟁 (수율 저하)
    R-W2: 여러 Lys에 DOTA 다중 부착 → stoichiometry 과잉 (방사화학 정제 복잡)
    R-W3: DOTA + N-term 점유 mod 동시 (WARNING — 합성 가능하나 효율 저하)

step08_stability 관련 N-methyl/lactam multiplier는 이번 범위에서 제외.
  → surrogate 제약으로 보류 (step08_stability 수정 금지 경계)

문헌:
  - Maecke HR et al. (2005) J Nucl Med 46:151S-159S  [DOTA-ε-amine coupling]
  - de Jong M et al. (2002) J Nucl Med 43:1650-1656   [DOTATATE N-terminus]
  - Liu S (2004) Chem Soc Rev 33:445-461              [bifunctional chelators]
  - Eberle AN et al. (2011) Curr Pharm Des 17:2974    [peptide modification sites]

HEURISTIC 경고 (H-06 가드):
  이 모듈은 규칙 기반 1차 필터이며 wet-lab 합성 결과를 보장하지 않음.
  MALDI-TOF, HPLC-RP, 방사화학 yield 실측으로 검증 필요.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 상수 및 타입 정의
# ---------------------------------------------------------------------------

# 수용 가능한 mod 접두어 (표기 규칙)
# 형식: "<mod_type>@<AA><position>" (1-indexed)
# 예: "fatty_acid@K5", "DOTA@K9", "PEG@K5", "D_aa@F7", "N-term_cap"
_MOD_PATTERN = re.compile(
    r"^(?P<mod_type>[A-Za-z0-9_\-]+)@(?P<aa>[A-Za-z])(?P<pos>\d+)$",
    re.IGNORECASE,
)
_NTERM_MODS = frozenset({
    "n-term_cap", "n_term_cap", "acetyl", "peg_n", "n-capping",
    "n_capping", "nterm_cap", "fmoc",
})
_DOTA_MOD_NAMES = frozenset({"dota", "nota", "nodaga", "dtpa", "dota_nhs"})
_FATTY_ACID_MOD_NAMES = frozenset({"fatty_acid", "c18", "c16", "acyl", "palmitoyl", "myristoyl"})
_PEG_MOD_NAMES = frozenset({"peg", "peg2", "peg4", "peg8", "mini_peg"})
_ACYLATION_MOD_NAMES = _FATTY_ACID_MOD_NAMES | _PEG_MOD_NAMES | frozenset({"acylation"})


@dataclass
class ConflictEntry:
    """단일 충돌 항목."""
    rule: str          # 규칙 코드 예: "R-F1"
    severity: str      # "FAIL" | "WARN"
    description: str   # 충돌 상세 설명
    mods_involved: List[str] = field(default_factory=list)  # 관련 mod 문자열


# ---------------------------------------------------------------------------
# mod 파싱 유틸리티
# ---------------------------------------------------------------------------


def _parse_mod(mod_str: str) -> Optional[Dict[str, Any]]:
    """mod 문자열을 파싱하여 mod_type, aa, position(1-indexed) 반환.

    지원 형식:
      - "<mod_type>@<AA><pos>"  예: "fatty_acid@K5", "DOTA@K9"
      - N-term 단독 표기        예: "N-term_cap", "acetyl"
      - Chelator 단독 표기      예: "DOTA", "NOTA" → N-terminus 부착 의도로 해석
        (위치 미지정 chelator = N-term 부착, de Jong 2002 / DOTATATE 선례)

    Returns:
        {"mod_type": str, "aa": str|None, "position": int|None, "raw": str}
        또는 파싱 불가 시 None.
    """
    raw = mod_str.strip()
    lower = raw.lower()

    # N-term 점유 mod (@ 없는 경우)
    if lower in _NTERM_MODS:
        return {"mod_type": lower, "aa": None, "position": None, "raw": raw}

    # Chelator 단독 표기 (@ 없음) → N-terminus 부착 의도
    # 예: "DOTA", "NOTA", "NODAGA", "DTPA"
    if lower in _DOTA_MOD_NAMES:
        return {"mod_type": lower, "aa": None, "position": None, "raw": raw}

    # "@" 포함 형식 파싱
    m = _MOD_PATTERN.match(raw)
    if m:
        return {
            "mod_type": m.group("mod_type").lower(),
            "aa": m.group("aa").upper(),
            "position": int(m.group("pos")),
            "raw": raw,
        }

    # 파싱 실패 — unknown 형식은 무시 (보수적)
    return None


def _is_mod_type(mod_type: str, name_set: frozenset) -> bool:
    """mod_type 문자열이 name_set 중 하나와 prefix 매칭."""
    for name in name_set:
        if mod_type.startswith(name) or mod_type == name:
            return True
    return False


def _is_dota(mod_type: str) -> bool:
    return _is_mod_type(mod_type, _DOTA_MOD_NAMES)


def _is_fatty_acid(mod_type: str) -> bool:
    return _is_mod_type(mod_type, _FATTY_ACID_MOD_NAMES)


def _is_peg(mod_type: str) -> bool:
    return _is_mod_type(mod_type, _PEG_MOD_NAMES)


def _is_acylation(mod_type: str) -> bool:
    return _is_mod_type(mod_type, _ACYLATION_MOD_NAMES)


def _is_nterm_occupying(mod_type: str) -> bool:
    """N-terminal amine을 점유하는 modification 여부."""
    return mod_type in _NTERM_MODS or mod_type.startswith("acetyl") or mod_type.startswith("peg_n")


# ---------------------------------------------------------------------------
# 충돌 규칙 검사 함수들
# ---------------------------------------------------------------------------


def _check_r_f1(
    parsed: List[Dict[str, Any]],
    sequence: str,
) -> List[ConflictEntry]:
    """R-F1: 동일 Lys에 fatty_acid + DOTA 동시.

    단일 ε-amine 경쟁 — 물리적으로 두 modification이 동시에 부착 불가.
    문헌: Maecke 2005 J Nucl Med — DOTA-NHS ε-amine coupling 단일성.
    """
    conflicts: List[ConflictEntry] = []

    # Lys 위치별 mod 그룹화
    lys_mods: Dict[int, List[Dict[str, Any]]] = {}
    for p in parsed:
        if p["aa"] == "K" and p["position"] is not None:
            pos = p["position"]
            lys_mods.setdefault(pos, []).append(p)

    for pos, mods in lys_mods.items():
        has_fa = any(_is_fatty_acid(m["mod_type"]) for m in mods)
        has_dota = any(_is_dota(m["mod_type"]) for m in mods)
        if has_fa and has_dota:
            involved = [m["raw"] for m in mods]
            conflicts.append(ConflictEntry(
                rule="R-F1",
                severity="FAIL",
                description=(
                    f"K{pos}에 fatty_acid와 DOTA 동시 부착 불가 — 단일 ε-amine 충돌. "
                    "DOTA-NHS ε-amine coupling은 단일 반응이므로 다른 acylation과 공존 불가 "
                    "(Maecke 2005 J Nucl Med)."
                ),
                mods_involved=involved,
            ))

    return conflicts


def _check_r_f2(
    parsed: List[Dict[str, Any]],
    sequence: str,
) -> List[ConflictEntry]:
    """R-F2: D-aa 치환 위치 + acylation 동일 위치.

    [BUG-FIX 버그4] FAIL → WARN으로 완화:
    D-Lys의 ε-amine NHS-ester 반응은 Cα 키랄성과 독립적 (Eberle 2011).
    GLP-1 세마글루타이드(L-Lys26 on-resin C18 acylation) 선례 존재; ε-amine은 Cα 키랄성 독립.
    → 합성 가능하나 chirality-HPLC 검증 필요 조건으로 경고 처리.

    D-aa 치환은 주쇄 Cα 배향을 반전시켜 side-chain ε-amine 접근성이 달라질 수 있음.
    acylation(fatty_acid, PEG)이 동일 잔기에 지정된 경우 반응성 저하 가능성 경고.
    단, Lys ε-amine의 NHS-ester 반응은 Cα 키랄성 독립: D-Lys도 acylation 가능.
    """
    conflicts: List[ConflictEntry] = []

    # D-aa 위치 수집
    d_aa_positions: Dict[int, str] = {}  # 1-indexed pos → AA
    for p in parsed:
        mt = p["mod_type"]
        if mt.startswith("d_") or mt.startswith("d-") or mt == "d_aa":
            if p["position"] is not None:
                d_aa_positions[p["position"]] = p.get("aa", "?")

    # acylation 위치 수집
    acyl_positions: Dict[int, List[Dict[str, Any]]] = {}
    for p in parsed:
        if _is_acylation(p["mod_type"]) and p["position"] is not None:
            acyl_positions.setdefault(p["position"], []).append(p)

    for pos in set(d_aa_positions) & set(acyl_positions):
        aa = d_aa_positions[pos]
        involved = (
            [p["raw"] for p in parsed if p.get("position") == pos and
             (p["mod_type"].startswith("d_") or p["mod_type"].startswith("d-"))]
            + [m["raw"] for m in acyl_positions[pos]]
        )
        conflicts.append(ConflictEntry(
            rule="R-F2",
            severity="WARN",  # [BUG-FIX 버그4] FAIL → WARN (D-Lys NHS-ester 가능)
            description=(
                f"위치 {pos}({aa})에 D-aa 치환과 acylation 동시 지정 — "
                "D-Lys ε-amine의 NHS-ester 반응은 Cα 키랄성 독립적으로 가능 "
                "(세마글루타이드 L-Lys26 on-resin acylation 선례; ε-amine은 Cα 키랄성 독립; "
                "Eberle 2011 Curr Pharm Des). "
                "합성 후 chirality-HPLC 검증으로 부분 라세미화 여부 확인 필요."
            ),
            mods_involved=involved,
        ))

    return conflicts


def _check_r_f3(
    parsed: List[Dict[str, Any]],
    sequence: str,
) -> List[ConflictEntry]:
    """R-F3: N-term capping + DOTA N-term 동시.

    N-terminus를 점유하는 capping(acetyl, PEG_N 등)과
    DOTA의 N-terminus 부착이 동시에 지정된 경우.
    """
    conflicts: List[ConflictEntry] = []

    # N-term 점유 mod 탐지
    nterm_cap_mods: List[Dict[str, Any]] = [
        p for p in parsed
        if _is_nterm_occupying(p["mod_type"]) and p["position"] is None
    ]
    # DOTA에 position이 없으면 N-terminus 부착 의도로 판단
    dota_nterm_mods: List[Dict[str, Any]] = [
        p for p in parsed
        if _is_dota(p["mod_type"]) and p["position"] is None
    ]

    if nterm_cap_mods and dota_nterm_mods:
        involved = [m["raw"] for m in nterm_cap_mods + dota_nterm_mods]
        conflicts.append(ConflictEntry(
            rule="R-F3",
            severity="FAIL",
            description=(
                "N-terminus를 점유하는 capping modification과 DOTA N-terminus 부착이 동시에 지정됨. "
                "N-terminus α-NH₂는 단일 반응 site — 두 modification 공존 불가 "
                "(de Jong 2002 J Nucl Med; Maecke 2005 J Nucl Med)."
            ),
            mods_involved=involved,
        ))

    return conflicts


def _check_r_f4(
    parsed: List[Dict[str, Any]],
    sequence: str,
) -> List[ConflictEntry]:
    """R-F4: DOTA가 FWKT pharmacophore 구성 Lys(K9, 1-indexed=9)에 부착.

    [버그4 신규 규칙] DOTA@FWKT-Lys → FAIL (생물활성 소멸)
    K9(Lys9)은 FWKT pharmacophore의 Lys 잔기로 SSTR2 pocket 직접 접촉 (PDB 7T11).
    Reubi 2017: K9→Ala 변이 시 Ki ×50배 상승 → DOTA 부착으로도 동일한 결합 방해 예상.

    SST-14 기준 FWKT-Lys 위치: 1-indexed=9 (0-indexed=8, seq[8]='K')
    서열이 다른 펩타이드는 FWKT 모티프 내 모든 Lys를 동일하게 처리.

    문헌:
      - Reubi JC et al. (2017) J Nucl Med 58:1017-1023 [K9→Ala Ki ×50]
      - Zhao X et al. (2022) Nature 605:204-209 [PDB 7T11 — SSTR2 cryo-EM + octreotide]
    """
    # FWKT 모티프 내 Lys 위치 탐색 (서열 기반)
    seq_upper = sequence.upper()
    fwkt_lys_positions: List[int] = []  # 1-indexed
    fwkt_idx = seq_upper.find("FWKT")
    if fwkt_idx >= 0:
        # FWKT 모티프 내 K 위치 (T는 아님 — K는 FWKT[2]='K')
        k_in_fwkt = fwkt_idx + 2  # 0-indexed K 위치
        fwkt_lys_positions.append(k_in_fwkt + 1)  # 1-indexed

    if not fwkt_lys_positions:
        return []  # FWKT 없는 서열은 이 규칙 해당 없음

    conflicts: List[ConflictEntry] = []
    for p in parsed:
        if _is_dota(p["mod_type"]) and p["position"] in fwkt_lys_positions:
            conflicts.append(ConflictEntry(
                rule="R-F4",
                severity="FAIL",
                description=(
                    f"DOTA가 FWKT pharmacophore 구성 Lys(위치 {p['position']})에 부착됨 — "
                    "생물활성 소멸 위험. K9는 SSTR2 pocket 직접 접촉 잔기이며 "
                    "K9→Ala 변이 시 Ki ×50 상승 (Reubi 2017 J Nucl Med). "
                    "DOTA 부착으로 동일한 결합 방해 예상. "
                    "DOTA는 N-terminus 또는 FWKT 외부 Lys(K4 등)에 부착 권장 "
                    "(DOTATATE 선례: de Jong 2002 J Nucl Med; Maecke 2005 J Nucl Med)."
                ),
                mods_involved=[p["raw"]],
            ))

    return conflicts


def _check_r_w1(
    parsed: List[Dict[str, Any]],
    sequence: str,
) -> List[ConflictEntry]:
    """R-W1: 동일 Lys에 fatty_acid + PEG 동시 (WARN).

    ε-amine 경쟁 — 기술적으로 가능하나 수율 저하 우려.
    """
    conflicts: List[ConflictEntry] = []

    lys_mods: Dict[int, List[Dict[str, Any]]] = {}
    for p in parsed:
        if p["aa"] == "K" and p["position"] is not None:
            lys_mods.setdefault(p["position"], []).append(p)

    for pos, mods in lys_mods.items():
        has_fa = any(_is_fatty_acid(m["mod_type"]) for m in mods)
        has_peg = any(_is_peg(m["mod_type"]) for m in mods)
        if has_fa and has_peg:
            involved = [m["raw"] for m in mods]
            conflicts.append(ConflictEntry(
                rule="R-W1",
                severity="WARN",
                description=(
                    f"K{pos}에 fatty_acid와 PEG 동시 지정 — ε-amine 경쟁으로 수율 저하 우려. "
                    "별도 직교 반응(orthogonal chemistry) 설계 권장."
                ),
                mods_involved=involved,
            ))

    return conflicts


def _check_r_w2(
    parsed: List[Dict[str, Any]],
    sequence: str,
) -> List[ConflictEntry]:
    """R-W2: 여러 Lys 또는 위치에 DOTA 다중 부착 (WARN).

    stoichiometry 과잉 — 방사화학 정제 복잡 및 비특이적 결합 우려.
    """
    dota_sites: List[Dict[str, Any]] = [
        p for p in parsed if _is_dota(p["mod_type"])
    ]
    if len(dota_sites) >= 2:
        involved = [m["raw"] for m in dota_sites]
        return [ConflictEntry(
            rule="R-W2",
            severity="WARN",
            description=(
                f"DOTA {len(dota_sites)}개 위치 동시 지정 — "
                "stoichiometry 과잉으로 방사화학 정제 복잡 및 비특이적 결합 우려. "
                "1:1 stoichiometry 표준 권장 (Liu 2004 Chem Soc Rev)."
            ),
            mods_involved=involved,
        )]
    return []


def _check_r_w3(
    parsed: List[Dict[str, Any]],
    sequence: str,
) -> List[ConflictEntry]:
    """R-W3: DOTA(위치 지정) + N-term 점유 mod 동시 (WARN).

    DOTA가 Lys에 지정되고 N-term이 capping된 경우:
    N-term DOTA 대안이 차단되어 라벨링 효율에 영향 가능.
    (합성 자체는 가능 — FAIL이 아닌 WARN)
    """
    conflicts: List[ConflictEntry] = []

    # Lys에 DOTA 지정된 경우
    dota_on_lys: List[Dict[str, Any]] = [
        p for p in parsed if _is_dota(p["mod_type"]) and p["aa"] == "K"
    ]
    nterm_cap_mods: List[Dict[str, Any]] = [
        p for p in parsed if _is_nterm_occupying(p["mod_type"])
    ]

    if dota_on_lys and nterm_cap_mods:
        involved = [m["raw"] for m in dota_on_lys + nterm_cap_mods]
        conflicts.append(ConflictEntry(
            rule="R-W3",
            severity="WARN",
            description=(
                "DOTA가 Lys에 지정되고 N-terminus가 capping됨 — "
                "DOTATATE 선례의 N-terminus DOTA 대안이 차단됨. "
                "합성은 가능하나 라벨링 효율 저하 가능성. 검증 권장."
            ),
            mods_involved=involved,
        ))

    return conflicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_modification_conflicts(
    sequence: str,
    mod_list: List[str],
) -> Dict[str, Any]:
    """SST-14 유사 펩타이드의 modification 충돌 검사.

    step08_stability 수정 없이 독립적으로 동작하는 순수 규칙 기반 검사기.

    Args:
        sequence: 1-letter 아미노산 서열. 예: "AGCKNFFWKTFTSC"
        mod_list: modification 목록.
                  형식: ["<mod_type>@<AA><pos_1idx>", ...]
                  예:   ["fatty_acid@K5", "DOTA@K9", "PEG@K5", "N-term_cap"]

    Returns:
        {
          "ok": bool,                  # True이면 FAIL 없음 (WARN은 있을 수 있음)
          "conflicts": List[dict],     # 충돌 목록 (빈 리스트이면 충돌 없음)
          "fail_count": int,           # FAIL 규칙 발동 횟수
          "warn_count": int,           # WARN 규칙 발동 횟수
          "sequence": str,             # 입력 서열 (echo)
          "mod_list": List[str],       # 입력 mod 목록 (echo)
        }

    충돌 없음 예시 (SST-14 WT + N-term DOTA):
        check_modification_conflicts("AGCKNFFWKTFTSC", ["DOTA"])
        → {"ok": True, "conflicts": [], "fail_count": 0, "warn_count": 0, ...}

    FAIL 예시 (동일 Lys에 fatty_acid+DOTA):
        check_modification_conflicts("AGCKNFFWKTFTSC", ["fatty_acid@K5", "DOTA@K5"])
        → {"ok": False, "conflicts": [{"rule": "R-F1", "severity": "FAIL", ...}], ...}

    WARN 예시 (동일 Lys에 fatty_acid+PEG):
        check_modification_conflicts("AGCKNFFWKTFTSC", ["fatty_acid@K5", "PEG@K5"])
        → {"ok": True, "conflicts": [{"rule": "R-W1", "severity": "WARN", ...}], ...}

    surrogate 제약으로 보류된 항목:
        - N-methyl/lactam multiplier (step08_stability.py 경계)
          → step08_stability 수정 금지 경계로 이번 범위에서 제외

    HEURISTIC 경고:
        이 모듈은 규칙 기반 1차 필터. wet-lab 결과 보장 안 함.
        MALDI-TOF, HPLC-RP, 방사화학 yield 실측 필요.
    """
    if not sequence:
        return {
            "ok": False,
            "conflicts": [{"rule": "INPUT", "severity": "FAIL",
                           "description": "서열이 비어 있음.", "mods_involved": []}],
            "fail_count": 1,
            "warn_count": 0,
            "sequence": sequence,
            "mod_list": mod_list,
        }

    seq_upper = sequence.upper().strip()

    # mod 파싱
    parsed: List[Dict[str, Any]] = []
    for mod_str in mod_list:
        result = _parse_mod(mod_str)
        if result is not None:
            parsed.append(result)

    # 시퀀스 길이 검증: 1-indexed position이 서열 길이를 초과하는지 확인
    invalid_pos_conflicts: List[ConflictEntry] = []
    for p in parsed:
        if p["position"] is not None and p["position"] > len(seq_upper):
            invalid_pos_conflicts.append(ConflictEntry(
                rule="R-INPUT",
                severity="FAIL",
                description=(
                    f"mod '{p['raw']}'의 위치 {p['position']}이 서열 길이 "
                    f"{len(seq_upper)}를 초과함."
                ),
                mods_involved=[p["raw"]],
            ))

    # AA 불일치 검증: 지정된 AA가 서열의 해당 위치 AA와 일치하는지
    mismatch_conflicts: List[ConflictEntry] = []
    for p in parsed:
        if p["aa"] is not None and p["position"] is not None:
            pos_0idx = p["position"] - 1
            if 0 <= pos_0idx < len(seq_upper):
                actual_aa = seq_upper[pos_0idx]
                if actual_aa != p["aa"]:
                    mismatch_conflicts.append(ConflictEntry(
                        rule="R-INPUT",
                        severity="WARN",
                        description=(
                            f"mod '{p['raw']}': 위치 {p['position']}의 잔기가 "
                            f"'{actual_aa}'이지만 mod는 '{p['aa']}'를 기대함. "
                            "서열과 mod_list 불일치 — 검토 필요."
                        ),
                        mods_involved=[p["raw"]],
                    ))

    # 충돌 규칙 실행
    all_conflicts: List[ConflictEntry] = (
        invalid_pos_conflicts
        + mismatch_conflicts
        + _check_r_f1(parsed, seq_upper)
        + _check_r_f2(parsed, seq_upper)   # [BUG-FIX 버그4] WARN으로 완화
        + _check_r_f3(parsed, seq_upper)
        + _check_r_f4(parsed, seq_upper)   # [버그4 신규] DOTA@FWKT-Lys → FAIL
        + _check_r_w1(parsed, seq_upper)
        + _check_r_w2(parsed, seq_upper)
        + _check_r_w3(parsed, seq_upper)
    )

    fail_count = sum(1 for c in all_conflicts if c.severity == "FAIL")
    warn_count = sum(1 for c in all_conflicts if c.severity == "WARN")

    return {
        "ok": fail_count == 0,
        "conflicts": [
            {
                "rule": c.rule,
                "severity": c.severity,
                "description": c.description,
                "mods_involved": c.mods_involved,
            }
            for c in all_conflicts
        ],
        "fail_count": fail_count,
        "warn_count": warn_count,
        "sequence": sequence,
        "mod_list": mod_list,
    }
