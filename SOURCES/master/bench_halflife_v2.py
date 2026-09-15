#!/usr/bin/env python3
"""반감기 벤치마크 v2 — L-aa vs D-aa 분리 + 메타데이터 회복. bio-tools python 실행.

입력: BENCH_halflife_v2.md (aa_type, modifications 필드). 없으면 BENCH_literature.md 폴백.
출력: BENCH_halflife_v2_results.json — 그룹별(ALL/L/D) raw·with-metadata Spearman.
할루시네이션 견제: 문헌값=출처 인용 JSON, 예측=실제 코드 실행.
"""
import sys, json, re, os
sys.path.insert(0,"[LOCAL_PATH]")
A="[LOCAL_PATH]"
KNOWN={"fatty_acid","pegylation","d_amino_acid","cyclization","substitution"}

def load():
    for fn in ("BENCH_halflife_v2.md","BENCH_literature.md"):
        p=f"{A}/{fn}"
        if os.path.exists(p):
            m=re.search(r"```json\s*(.*?)```",open(p,encoding="utf-8").read(),re.DOTALL)
            if m: return json.loads(m.group(1)), fn
    raise SystemExit("벤치마크 JSON 없음")

def spearman(xs,ys):
    p=[(x,y) for x,y in zip(xs,ys) if isinstance(x,(int,float)) and isinstance(y,(int,float))]; n=len(p)
    if n<3: return None,n
    def rk(v):
        idx=sorted(range(len(v)),key=lambda i:v[i]); r=[0.0]*len(v); i=0
        while i<len(v):
            j=i
            while j+1<len(v) and v[idx[j+1]]==v[idx[i]]: j+=1
            for k in range(i,j+1): r[idx[k]]=(i+j)/2+1
            i=j+1
        return r
    a=[q[0] for q in p]; b=[q[1] for q in p]; ra,rb=rk(a),rk(b); n=len(p)
    ma,mb=sum(ra)/n,sum(rb)/n
    cov=sum((x-ma)*(y-mb) for x,y in zip(ra,rb)); va=sum((x-ma)**2 for x in ra)**.5; vb=sum((y-mb)**2 for y in rb)**.5
    return (round(cov/(va*vb),3) if va*vb else None),n

def main():
    data,src=load()
    from pyrosetta_flow.halflife_ensemble import ensemble_halflife
    from AG_src.pipeline.step08_stability import predict_half_life
    rows=[]
    for d in data:
        seq=(d.get("sequence") or "").upper()
        if not seq or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
            rows.append({**d,"_skip":True}); continue
        mods=[m for m in (d.get("modifications") or []) if m in KNOWN]
        raw=ensemble_halflife(seq).get("half_life_h")
        meta=predict_half_life(seq, mods)
        # aa_type 정규화
        at=(d.get("aa_type") or "").upper()
        grp = "D" if at in ("D","MIXED") or "d_amino_acid" in (d.get("modifications") or []) else "L"
        rows.append({"name":d.get("name"),"sequence":seq,"lit":d.get("half_life_h"),
                     "raw":raw,"meta":meta,"group":grp,"mods":mods,"src":d.get("half_life_src")})
    R=[r for r in rows if not r.get("_skip")]
    def grp(g): return [r for r in R if r["group"]==g]
    out={"source":src,"n":len(R),
         "all_raw":spearman([r["lit"] for r in R],[r["raw"] for r in R]),
         "all_meta":spearman([r["lit"] for r in R],[r["meta"] for r in R]),
         "L_raw":spearman([r["lit"] for r in grp("L")],[r["raw"] for r in grp("L")]),
         "L_meta":spearman([r["lit"] for r in grp("L")],[r["meta"] for r in grp("L")]),
         "D_raw":spearman([r["lit"] for r in grp("D")],[r["raw"] for r in grp("D")]),
         "D_meta":spearman([r["lit"] for r in grp("D")],[r["meta"] for r in grp("D")]),
         "n_L":len(grp("L")),"n_D":len(grp("D")),"rows":R}
    json.dump(out,open(f"{A}/BENCH_halflife_v2_results.json","w"),ensure_ascii=False,indent=2)
    print(f"src={src} n={len(R)} (L={out['n_L']} D={out['n_D']})")
    print(f"ALL raw={out['all_raw']} meta={out['all_meta']}")
    print(f"L   raw={out['L_raw']} meta={out['L_meta']}")
    print(f"D   raw={out['D_raw']} meta={out['D_meta']}")

if __name__=="__main__": main()
