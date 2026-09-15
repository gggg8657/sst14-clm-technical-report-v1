"""half-life RF 모델용 feature 추출 — train_halflife_features.py 의 featurize 와 **동일**해야 함.
(feat_cols 순서/정의가 학습 시점과 일치해야 joblib 모델이 올바르게 동작.)
2026-06-09 C. 학습 스크립트: _workspace/pepmsnd_local/scripts/train_halflife_features.py
"""
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
AAS = list("ACDEFGHIKLMNPQRSTVWY")


def featurize(seq, rec):
    seq = (seq or "").upper()
    seq = "".join(c for c in seq if c in AAS)  # 표준20만
    n = len(seq)
    if n < 2:
        return None
    rec = rec or {}
    f = {}
    f["length"] = n
    for a in AAS:
        f[f"frac_{a}"] = seq.count(a) / n
    f["gravy"] = sum(KD[a] for a in seq) / n
    f["net_charge"] = (seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
                       - seq.count("D") - seq.count("E"))
    f["n_arg_lys"] = seq.count("R") + seq.count("K")
    f["n_met"] = seq.count("M")
    f["n_cys"] = seq.count("C")
    f["n_pro"] = seq.count("P")
    cleav = 0.0
    for i in range(n - 1):
        if seq[i] in "KR" and seq[i + 1] != "P":
            cleav += 1.0
        elif seq[i] in "FYW" and seq[i + 1] != "P":
            cleav += 0.5
    f["cleavage_sites"] = cleav
    f["cleavage_density"] = cleav / n
    f["nterm_KR"] = 1.0 if seq[0] in "KR" else 0.0
    chiral = str(rec.get("chiral", "L"))
    f["has_D"] = 1.0 if chiral in ("D", "Mix", "DL") else 0.0
    f["is_cyclic"] = 1.0 if str(rec.get("lin_cyc", "")).lower().startswith("cyc") else (
        1.0 if seq.count("C") >= 2 else 0.0)
    f["chem_mod"] = 0.0 if str(rec.get("chem_mod", "None")).strip() in ("None", "", "N.A.") else 1.0
    nter = str(rec.get("nter", "Free")).strip().lower()
    cter = str(rec.get("cter", "Free")).strip().lower()
    f["nter_capped"] = 0.0 if nter in ("free", "none", "n.a.", "") else 1.0
    f["cter_capped"] = 0.0 if cter in ("free", "none", "n.a.", "") else 1.0
    cm = str(rec.get("chem_mod", "")).lower()
    f["lipid_or_peg"] = 1.0 if any(k in cm for k in ("lipid", "fatty", "palmit", "myrist", "peg", "acyl", "stear")) else 0.0
    return f
