#!/usr/bin/env python3
"""벤치마크 러너 — 문헌 ground-truth(BENCH_literature.md JSON) vs 우리 예측 순위.

할루시네이션 견제: 문헌값은 researcher가 출처 인용으로 수집한 JSON에서만, 우리값은 실제 코드 실행.
출력: _analysis/BENCH_results.md (표 + Spearman/Kendall + 정성 경향성). bio-tools python 으로 실행.
"""
import sys, json, re, os
sys.path.insert(0, "[LOCAL_PATH]")
ANALYSIS = "[LOCAL_PATH]"

def load_bench():
    txt = open(f"{ANALYSIS}/BENCH_literature.md", encoding="utf-8").read()
    m = re.search(r"```json\s*(.*?)```", txt, re.DOTALL)
    if not m: raise SystemExit("BENCH_literature.md 에 ```json 블록 없음")
    return json.loads(m.group(1))

def spearman(xs, ys):
    """동시 존재하는 쌍만으로 Spearman ρ. n<3 이면 None."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3: return None, n
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v)
        i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[idx[j+1]] == v[idx[i]]: j += 1
            avg = (i+j)/2.0 + 1
            for k in range(i, j+1): r[idx[k]] = avg
            i = j+1
        return r
    xs2 = [p[0] for p in pairs]; ys2 = [p[1] for p in pairs]
    rx, ry = rank(xs2), rank(ys2)
    mx, my = sum(rx)/n, sum(ry)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    vx = sum((a-mx)**2 for a in rx)**0.5; vy = sum((b-my)**2 for b in ry)**0.5
    return (cov/(vx*vy) if vx*vy else None), n

def main():
    data = load_bench()
    from pyrosetta_flow.halflife_ensemble import ensemble_halflife
    from pyrosetta_flow.multiobjective import cheap_objectives, predict_toxicity_for_sequences

    seqs = [d.get("sequence","") for d in data]
    # pepADMET 독성(hc50) 배치 (subprocess; 실패 시 빈 dict)
    try:
        tox = predict_toxicity_for_sequences([s for s in seqs if s])
    except Exception as e:
        tox = {}; print("tox 실패:", e, file=sys.stderr)

    rows = []
    for d in data:
        seq = d.get("sequence","")
        if not seq: continue
        rec = {"name": d.get("name"), "sequence": seq,
               "lit_hl": d.get("half_life_h"), "lit_hl_src": d.get("half_life_src"),
               "lit_hc50": d.get("hc50"), "modified": d.get("modified"), "note": d.get("note","")}
        try:
            ens = ensemble_halflife(seq)
            rec["our_hl"] = ens.get("ensemble_hours") or ens.get("half_life_h") or ens.get("hours")
            rec["our_hl_raw"] = ens
        except Exception as e:
            rec["our_hl"] = None; rec["err_hl"] = str(e)
        try:
            co = cheap_objectives(seq)
            rec["our_admet"] = co.get("admet_score")
            rec["our_hl2"] = co.get("half_life_h") or co.get("ensemble_halflife")
        except Exception as e:
            rec["our_admet"] = None
        t = tox.get(seq, {})
        rec["our_hc50"] = t.get("hc50"); rec["our_toxic"] = t.get("is_toxic")
        rows.append(rec)

    # 우리 반감기 추정치 통일 (ensemble 우선)
    for r in rows:
        r["our_hl_final"] = r.get("our_hl") if r.get("our_hl") is not None else r.get("our_hl2")

    rho_hl, n_hl = spearman([r["lit_hl"] for r in rows], [r["our_hl_final"] for r in rows])
    rho_tox, n_tox = spearman([r["lit_hc50"] for r in rows], [r["our_hc50"] for r in rows])
    # 변형 제외(plain) 반감기 상관
    plain = [r for r in rows if not r.get("modified")]
    rho_hl_plain, n_plain = spearman([r["lit_hl"] for r in plain], [r["our_hl_final"] for r in plain])

    json.dump({"rows": rows, "rho_hl": rho_hl, "n_hl": n_hl,
               "rho_hl_plain": rho_hl_plain, "n_plain": n_plain,
               "rho_tox": rho_tox, "n_tox": n_tox},
              open(f"{ANALYSIS}/BENCH_results.json","w"), ensure_ascii=False, indent=2)
    print(f"DONE rho_hl={rho_hl} (n={n_hl}) | plain={rho_hl_plain}(n={n_plain}) | tox={rho_tox}(n={n_tox}) | rows={len(rows)}")

if __name__ == "__main__":
    main()
