#!/usr/bin/env python3
"""ADMET 벤치마크 Part B — 용혈 독성: 측정 HC50/class vs pepADMET 예측.

문헌(BENCH_admet_literature.md JSON)의 측정 용혈값 vs 우리 pepADMET hc50/is_toxic.
- Spearman: 측정 HC50(µg/mL, 낮을수록 독성) vs 우리 hc50(음수일수록 독성) → 부호 보정해 상관.
- 분류 AUC: (-our_hc50)를 점수로 hemolytic(true/false) 판별 능력.
할루시네이션 견제: 문헌값=출처 인용 JSON, 우리값=pepADMET 실제 실행. bio-tools python 으로 실행.
"""
import sys, json, re
sys.path.insert(0, "[LOCAL_PATH]")
ANALYSIS = "[LOCAL_PATH]"

def load():
    t = open(f"{ANALYSIS}/BENCH_admet_literature.md", encoding="utf-8").read()
    m = re.search(r"```json\s*(.*?)```", t, re.DOTALL)
    if not m: raise SystemExit("BENCH_admet_literature.md JSON 블록 없음")
    return json.loads(m.group(1))

def spearman(xs, ys):
    p = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(p)
    if n < 3: return None, n
    def rk(v):
        idx = sorted(range(len(v)), key=lambda i: v[i]); r=[0.0]*len(v); i=0
        while i < len(v):
            j=i
            while j+1 < len(v) and v[idx[j+1]]==v[idx[i]]: j+=1
            for k in range(i,j+1): r[idx[k]]=(i+j)/2+1
            i=j+1
        return r
    a=[q[0] for q in p]; b=[q[1] for q in p]; ra,rb=rk(a),rk(b); n=len(p)
    ma,mb=sum(ra)/n,sum(rb)/n
    cov=sum((x-ma)*(y-mb) for x,y in zip(ra,rb)); va=sum((x-ma)**2 for x in ra)**.5; vb=sum((y-mb)**2 for y in rb)**.5
    return (cov/(va*vb) if va*vb else None), n

def auc(scores, labels):
    """라벨(1=hemolytic) 분류에서 score(높을수록 독성)의 ROC-AUC (Mann-Whitney)."""
    pos=[s for s,l in zip(scores,labels) if l]; neg=[s for s,l in zip(scores,labels) if l is False]
    if not pos or not neg: return None, len(pos), len(neg)
    wins=0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p>n else (0.5 if p==n else 0.0)
    return wins/(len(pos)*len(neg)), len(pos), len(neg)

def main():
    data = load()
    from pyrosetta_flow.multiobjective import predict_toxicity_for_sequences
    STD = set("ACDEFGHIKLMNPQRSTVWY")
    def valid(s): return s and all(c in STD for c in s.upper())
    # 비표준/사이클릭 서열은 배치 전체를 실패시키므로 제외(기록)
    seqs = [d["sequence"].upper() for d in data if valid(d.get("sequence",""))]
    excluded = [d.get("name") for d in data if not valid(d.get("sequence",""))]
    print(f"제외(비표준 서열): {excluded}", file=sys.stderr)
    # 가동 중 discovery 엔진의 pepADMET 호출과 충돌 가능 → 배치 후 누락분 개별 재시도(최대 3회)
    import time
    tox = {}
    try: tox = dict(predict_toxicity_for_sequences(seqs) or {})
    except Exception: pass
    for attempt in range(3):
        missing = [s for s in seqs if s not in tox or tox.get(s,{}).get("hc50") is None]
        if not missing: break
        print(f"  재시도 {attempt+1}: 누락 {len(missing)}", file=sys.stderr)
        for s in missing:
            try:
                r = predict_toxicity_for_sequences([s])
                if r and r.get(s,{}).get("hc50") is not None: tox[s]=r[s]
            except Exception: pass
            time.sleep(0.3)
    print(f"  최종 확보: {sum(1 for s in seqs if tox.get(s,{}).get('hc50') is not None)}/{len(seqs)}", file=sys.stderr)
    rows=[]
    for d in data:
        s=d.get("sequence","")
        if not s: continue
        t=tox.get(s.upper(),{})
        rows.append({"name":d.get("name"), "sequence":s,
                     "lit_hc50":d.get("hc50_ugml"), "lit_hemolytic":d.get("hemolytic"),
                     "our_hc50":t.get("hc50"), "our_toxic":t.get("is_toxic"),
                     "graph_note":t.get("graph_note")})
    # Spearman: 측정 HC50(낮을수록 독성) vs 우리 hc50(음수일수록 독성).
    # 동일 방향으로: 측정 -log? 단순히 측정 HC50 작을수록=독성, our_hc50 작을수록(더 음수)=독성 → 같은 방향
    lit=[r["lit_hc50"] for r in rows]; our=[r["our_hc50"] for r in rows]
    rho, n_rho = spearman(lit, our)
    # 분류 AUC: score=-our_hc50 (클수록 독성), label=lit_hemolytic
    sc=[(-r["our_hc50"] if r["our_hc50"] is not None else None) for r in rows]
    lab=[r["lit_hemolytic"] for r in rows]
    pairs=[(s,l) for s,l in zip(sc,lab) if s is not None and l is not None]
    a, npos, nneg = auc([p[0] for p in pairs],[p[1] for p in pairs]) if pairs else (None,0,0)
    out={"rows":rows,"spearman_hc50":rho,"n_spearman":n_rho,"auc_hemolytic":a,"n_pos":npos,"n_neg":nneg}
    json.dump(out, open(f"{ANALYSIS}/BENCH_admet_tox_results.json","w"), ensure_ascii=False, indent=2)
    print(f"DONE spearman(hc50)={rho} n={n_rho} | AUC(hemolytic)={a} pos={npos} neg={nneg} | rows={len(rows)}")

if __name__=="__main__":
    main()
