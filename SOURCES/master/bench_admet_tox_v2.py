#!/usr/bin/env python3
"""ADMET 용혈 벤치마크 v2 — N 확장 + L-aa vs D-aa 분리 AUC. bio-tools python 실행.
입력: BENCH_admet_v2.md (aa_type 필드). 없으면 BENCH_admet_literature.md 폴백.
출력: BENCH_admet_v2_results.json. 가동 엔진 충돌 대비 누락분 개별 재시도.
"""
import sys, json, re, os, time
sys.path.insert(0,"[LOCAL_PATH]")
A="[LOCAL_PATH]"
STD=set("ACDEFGHIKLMNPQRSTVWY")

def load():
    for fn in ("BENCH_admet_v2.md","BENCH_admet_literature.md"):
        p=f"{A}/{fn}"
        if os.path.exists(p):
            m=re.search(r"```json\s*(.*?)```",open(p,encoding="utf-8").read(),re.DOTALL)
            if m: return json.loads(m.group(1)), fn
    raise SystemExit("ADMET 벤치마크 JSON 없음")

def auc(scores,labels):
    pos=[s for s,l in zip(scores,labels) if l is True]; neg=[s for s,l in zip(scores,labels) if l is False]
    if not pos or not neg: return None,len(pos),len(neg)
    w=sum((1.0 if p>n else 0.5 if p==n else 0.0) for p in pos for n in neg)
    return round(w/(len(pos)*len(neg)),3),len(pos),len(neg)

def main():
    data,src=load()
    from pyrosetta_flow.multiobjective import predict_toxicity_for_sequences
    def valid(s): return s and all(c in STD for c in s.upper())
    seqs=[d["sequence"].upper() for d in data if valid(d.get("sequence",""))]
    tox={}
    try: tox=dict(predict_toxicity_for_sequences(seqs) or {})
    except Exception: pass
    for _ in range(3):
        miss=[s for s in seqs if tox.get(s,{}).get("hc50") is None]
        if not miss: break
        for s in miss:
            try:
                r=predict_toxicity_for_sequences([s])
                if r and r.get(s,{}).get("hc50") is not None: tox[s]=r[s]
            except Exception: pass
            time.sleep(0.3)
    rows=[]
    for d in data:
        s=(d.get("sequence") or "").upper()
        if not valid(s): continue
        t=tox.get(s,{})
        at=(d.get("aa_type") or "").upper()
        nm=(d.get("name") or ""); note=(d.get("note") or "")
        # aa_type 없으면 이름/노트로 D 추론 (예: 'D-Piscidin', 'D-Dermaseptin')
        if at in ("D","MIXED") or nm.strip().upper().startswith("D-") or "d-amino" in note.lower() or "D-아미노" in note:
            grp="D"
        else:
            grp="L"
        rows.append({"name":d.get("name"),"sequence":s,"lit_hemolytic":d.get("hemolytic"),
                     "lit_hc50":d.get("hc50_ugml"),"our_hc50":t.get("hc50"),"group":grp})
    def AUC(sub):
        sc=[-r["our_hc50"] if r["our_hc50"] is not None else None for r in sub]
        lb=[r["lit_hemolytic"] for r in sub]
        pairs=[(x,y) for x,y in zip(sc,lb) if x is not None and y in (True,False)]
        return auc([p[0] for p in pairs],[p[1] for p in pairs]) if pairs else (None,0,0)
    L=[r for r in rows if r["group"]=="L"]; D=[r for r in rows if r["group"]=="D"]
    out={"source":src,"n":len(rows),"got":sum(1 for r in rows if r["our_hc50"] is not None),
         "auc_all":AUC(rows),"auc_L":AUC(L),"auc_D":AUC(D),
         "n_L":len(L),"n_D":len(D),"rows":rows}
    json.dump(out,open(f"{A}/BENCH_admet_v2_results.json","w"),ensure_ascii=False,indent=2)
    print(f"src={src} n={len(rows)} got={out['got']} (L={len(L)} D={len(D)})")
    print(f"AUC all={out['auc_all']} | L={out['auc_L']} | D={out['auc_D']}")

if __name__=="__main__": main()
