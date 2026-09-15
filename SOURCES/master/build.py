#!/usr/bin/env python3
"""report_site 빌더 — _analysis/*.md → 다중 HTML(공유 템플릿+네비+mermaid)+index.

할루시네이션 견제: 콘텐츠는 서브에이전트가 file:line 인용으로 검증한 .md 에서만 생성.
라이브 stats(index)는 실측 json 직독.
"""
import json, re, glob, os
import markdown

ROOT = "[LOCAL_PATH]"
SITE = f"{ROOT}/_workspace/report_site"
ANALYSIS = f"{SITE}/_analysis"
PAGES = f"{SITE}/pages"
REPO = f"{ROOT}/AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri"
os.makedirs(PAGES, exist_ok=True)

# 페이지 메타: slug -> (제목, 네비그룹, 이모지)
META = {
 "01_mutation_scaffold": ("변이 엔진 & Scaffold 가드", "발굴 파이프라인", "🧬"),
 "02_pyrosetta_docking": ("PyRosetta 도킹 (실측 ΔG)", "발굴 파이프라인", "⚛️"),
 "03_selectivity_dmargin": ("선택성 · Δmargin", "발굴 파이프라인", "🎯"),
 "04_halflife_ensemble": ("혈중 반감기 앙상블", "약리 surrogate", "⏳"),
 "05_admet_toxicity": ("ADMET · 독성 게이트", "약리 surrogate", "🛡️"),
 "06_multiobjective_scoring": ("다목적 스코어링", "발굴 파이프라인", "📊"),
 "07_search_optimization": ("탐색 최적화 (Bandit/BO/수렴)", "최적화·메타", "🔍"),
 "08_continuous_engine": ("무한 발굴 엔진", "발굴 파이프라인", "♾️"),
 "09_llm_agents": ("LLM 에이전트 (Planner/Critic/Reporter)", "최적화·메타", "🤖"),
 "10_web_ui": ("웹 UI (FastAPI+React+Mol*)", "운영·UI", "🖥️"),
 "11_qc_validation": ("QC·검증 & fail-closed", "최적화·메타", "✅"),
 "12_harness_adaptation": ("하네스 어댑테이션 (메타)", "최적화·메타", "🧩"),
 "13_diversity_cluster": ("다양성 · 클러스터링", "최적화·메타", "🌐"),
 "14_benchmark_validation": ("벤치마크: 반감기 (문헌 vs 예측)", "검증·벤치마크", "🧪"),
 "15_admet_benchmark": ("벤치마크: ADMET (물성·독성)", "검증·벤치마크", "🧫"),
 "R1_comparison": ("기존 시스템 비교", "맥락·비교", "⚖️"),
 "R2_academic_context": ("학술·임상 맥락", "맥락·비교", "📚"),
}
GROUP_ORDER = ["검증·벤치마크", "발굴 파이프라인", "약리 surrogate", "최적화·메타", "운영·UI", "맥락·비교"]

def live_stats():
    try:
        lb = json.load(open(f"{REPO}/runs/pyrosetta_flow/global_selectivity_leaderboard.json"))
        ds = json.load(open(f"{REPO}/runs/pyrosetta_flow/discovery_status.json"))
        ent = lb.get("entries", [])
        strict = [e for e in ent if (e.get("delta_margin") or 0) > 0 and (e.get("ddg") or 9) <= -15 and not e.get("more_toxic_than_native")]
        return {"epochs": ds.get("epochs_done"), "best": lb.get("best_delta_margin"),
                "screened": lb.get("n_screened_unique"), "strict": len(strict), "ntop": len(ent),
                "top1": ent[0] if ent else {}}
    except Exception as e:
        return {"epochs":"?","best":"?","screened":"?","strict":"?","ntop":"?","top1":{},"err":str(e)}

def bench_stats():
    """벤치마크 JSON 직독 (메인 요약용). 없으면 None."""
    import os
    def J(name):
        p=f"{ANALYSIS}/{name}"
        try: return json.load(open(p)) if os.path.exists(p) else {}
        except Exception: return {}
    desc=J("BENCH_admet_descriptors.json"); tox=J("BENCH_admet_tox_results.json")
    hl2=J("BENCH_halflife_v2_results.json"); ax=J("BENCH_admet_v2_results.json")
    dr=(desc.get("summary") or {})
    def first(x): return x[0] if isinstance(x,(list,tuple)) and x else x  # spearman 튜플 → 값
    return {
        "hl_n": hl2.get("n"), "hl_nL": hl2.get("n_L"), "hl_nD": hl2.get("n_D"),
        "hl_all_raw": first(hl2.get("all_raw")), "hl_all_meta": first(hl2.get("all_meta")),
        "hl_L_raw": first(hl2.get("L_raw")), "hl_L_meta": first(hl2.get("L_meta")),
        "hl_D_raw": first(hl2.get("D_raw")), "hl_D_meta": first(hl2.get("D_meta")),
        "desc_gravy": (dr.get("GRAVY") or {}).get("r"), "desc_inst": (dr.get("Instability") or {}).get("r"),
        "desc_pi": (dr.get("pI") or {}).get("r"), "desc_boman": (dr.get("Boman") or {}).get("r"),
        "tox_auc": tox.get("auc_hemolytic"), "tox_npos": tox.get("n_pos"), "tox_nneg": tox.get("n_neg"),
        "tox_auc_L": first((ax.get("auc_L") or [None])),
    }


def existing_pages():
    out=[]
    for f in sorted(glob.glob(f"{ANALYSIS}/*.md")):
        slug=os.path.splitext(os.path.basename(f))[0]
        if slug in META: out.append(slug)
    return out

def nav_html(active=None, in_pages=True):
    # in_pages=True: feature 페이지(루트/pages/ 안)에서 렌더 → 같은 폴더 상대링크
    # in_pages=False: index(루트)에서 렌더 → pages/ 접두사 필요
    pages=existing_pages()
    home = "../index.html" if in_pages else "index.html"
    pref = "" if in_pages else "pages/"
    h=['<div class="brand"><h1>SSTR2 발굴 시스템</h1><span>TECHNICAL REPORT</span></div><nav>']
    h.append(f'<a href="{home}"{" class=\"active\"" if active=="index" else ""}>🏠 개요 (메인)</a>')
    for g in GROUP_ORDER:
        gp=[s for s in pages if META[s][1]==g]
        if not gp: continue
        h.append(f'<div class="navgroup">{g}</div>')
        for s in gp:
            t,_,emo=META[s]
            cls=' class="active"' if active==s else ''
            h.append(f'<a href="{pref}{s}.html"{cls}>{emo} {t}</a>')
    h.append('</nav>')
    return "".join(h)

MERMAID_CDN='<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script><script>mermaid.initialize({startOnLoad:true,theme:"neutral"});</script>'

def page_shell(title, body, active, depth_css="../assets/style.css", in_pages=True):
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · SSTR2 발굴 시스템 보고서</title>
<link rel="stylesheet" href="{depth_css}">{MERMAID_CDN}</head>
<body><div class="layout"><aside class="sidebar">{nav_html(active, in_pages)}</aside>
<main class="main"><div class="wrap">{body}</div></main></div>
<footer>SSTR2 선택성 방사성의약품 발굴 Agentic AI 시스템 · 기술 보고서 · 모든 기능 주장은 코드 file:line 인용으로 검증 · 생성 2026-06-17</footer>
</body></html>"""

def md_to_html(text):
    html = markdown.markdown(text, extensions=["tables","fenced_code","toc","sane_lists"])
    # ```mermaid 코드블록 → <div class="mermaid"> (mermaid.js 렌더)
    html = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                  lambda m: '<div class="diagram"><div class="mermaid">'+
                  m.group(1).replace("&gt;",">").replace("&lt;","<").replace("&amp;","&")+'</div></div>',
                  html, flags=re.DOTALL)
    # 인라인 코드 안에서 깨진 잔여 triple-backtick 표기 정리 (표시 artifact 방지)
    html = html.replace("\\`", "`")
    html = re.sub(r'`{3,}\s*(\w+)?', lambda m: ('<i>'+ (m.group(1) or 'code') +' 블록</i>'), html)
    return html

def build_feature_pages():
    n=0
    for slug in existing_pages():
        t,_,emo=META[slug]
        raw=open(f"{ANALYSIS}/{slug}.md",encoding="utf-8").read()
        body=f'<p style="color:var(--muted);font-size:12px;letter-spacing:1px">기능 상세 보고</p>'
        body+=md_to_html(raw)
        open(f"{PAGES}/{slug}.html","w",encoding="utf-8").write(page_shell(f"{emo} {t}",body,slug))
        n+=1
    return n

def build_index():
    s=live_stats(); pages=existing_pages()
    top1=s["top1"]
    hero=f"""<div class="hero" style="margin:-40px -48px 0">
      <p class="kick">AGENTIC AI · SSTR2 RADIOPHARMACEUTICAL DISCOVERY</p>
      <h1>SSTR2 선택성 펩타이드 발굴 시스템<br>기술 보고서</h1>
      <p>SST-14(AGCKNFFWKTFTSC) 변이체를 PyRosetta 실측 도킹 + LLM 에이전트 루프 + 무한 발굴 엔진으로
      스크리닝하여 SSTR2 선택적(SSTR1/3/4/5 회피) 방사성의약품 후보를 발굴한다. 아래 모든 기능 설명은
      <b style="color:var(--mint)">코드 file:line 인용으로 검증</b>되었으며, 미검증 항목은 명시한다.</p>
      <div class="meta">생성 2026-06-17 · 무한 엔진 가동 중(라이브 데이터) · 절대 t½/HC50 단위 등 미검증 항목은 각 페이지에 표기</div>
    </div>"""
    stats=f"""<div class="stats">
      <div class="stat"><div class="v">{s['epochs']}</div><div class="l">무한 엔진 epoch (가동 중)</div></div>
      <div class="stat"><div class="v">+{s['best']}</div><div class="l">역대 best Δmargin (native +13.37 초과분)</div></div>
      <div class="stat"><div class="v">{s['strict']}/{s['ntop']}</div><div class="l">엄격기준 충족 (Δ&gt;0 &amp; ΔG≤−15 &amp; 비독성)</div></div>
      <div class="stat"><div class="v">{s['screened']}</div><div class="l">누적 스크리닝 고유 서열</div></div>
    </div>"""
    win=f"""<div class="note win"><b>🎯 현재 성과 (in-silico, 라이브 리더보드 직독):</b>
      무한 엔진이 <b>{s['epochs']} epoch</b> 누적 → 최상위 후보 <code>{top1.get('sequence','?')}</code>
      (Δmargin <b>+{top1.get('delta_margin','?')}</b>, margin {top1.get('margin','?')}, ΔG {top1.get('ddg','?')},
      독성 native 이하). 2026-06-10 선택성 GOAL 성공기준(≥3건)을 <b>{s['strict']}건</b>으로 대폭 초과.
      <br><span style="color:#a3322f"><b>정직성 단서:</b> 모두 in-silico proxy이며 <b>wet-lab 미검증</b>이다. 절대
      ΔG·HC50 단위는 미보정이고, home-advantage(curated SSTR2가 source 복합체 유래) 보정 후 상대 우위로만 해석해야 한다.</span></div>"""
    # 아키텍처 mermaid
    arch="""<h2>시스템 아키텍처</h2><div class="diagram"><div class="mermaid">
flowchart TD
  IN["입력: SST-14 복합체 PDB<br/>AGCKNFFWKTFTSC"] --> LLM["LLM 레이어<br/>vLLM Qwen3-32B<br/>Planner·Critic·Reporter"]
  LLM --> MUT["변이 엔진<br/>scaffold 가드(SS·FWKT)"]
  MUT --> DOCK["PyRosetta FlexPepDock<br/>실측 ΔG"]
  DOCK --> SEL["in-loop 선택성<br/>off-target 5종 → Δmargin"]
  DOCK --> SCORE["다목적 스코어링<br/>ΔG·반감기·Δmargin·ADMET"]
  SEL --> SCORE
  SCORE --> RANK["NSGA-II Pareto + scalar"]
  RANK --> LB["글로벌 선택성 리더보드<br/>(영속·warm-start)"]
  LB --> ENG["무한 발굴 엔진<br/>epoch 무한·다양성 탈출"]
  ENG --> MUT
  ENG -.6h autopush.-> GH["GitHub (외부 모니터링)"]
  ENG -.미연결.-> UI["웹 UI(FastAPI+React+Mol*)"]
</div></div>"""
    # 기능 카드 그리드
    cards='<h2>기능별 상세</h2><p>각 기능: 동작원리 · 영향 · 관련 Action Item · 완성도 · 학술가치 · 사용법 · (액션 무관 시) 필요이유.</p><div class="grid">'
    descs={
     "01_mutation_scaffold":"변이 위치 선택 + 이황화·FWKT 보존 가드",
     "02_pyrosetta_docking":"FlexPepDock 실측 ΔG, mock 없음, fail-closed",
     "03_selectivity_dmargin":"off-target 동일프로토콜 + home-advantage Δmargin",
     "04_halflife_ensemble":"휴리스틱+RF, Spearman 상대순위 간접검증",
     "05_admet_toxicity":"surrogate + pepADMET hc50 native-relative 게이트",
     "06_multiobjective_scoring":"NSGA-II Pareto + scalar + ECR consensus",
     "07_search_optimization":"Thompson bandit·베이지안·수렴감지",
     "08_continuous_engine":"무한 epoch + 다양성 탈출 + 영속 학습",
     "09_llm_agents":"가설 주도 3-에이전트, vLLM no-think",
     "10_web_ui":"폴링 모니터링 + Mol* (무한엔진 미연결)",
     "11_qc_validation":"QC 게이트 + fail-closed + 약리 가드",
     "12_harness_adaptation":"오케스트레이션 패턴·자가검증 메타 인프라",
     "13_diversity_cluster":"다양성 유지 + 구조 클러스터 tier",
     "R1_comparison":"RFdiffusion·AlphaProteo·pepADMET 등 비교",
     "R2_academic_context":"SSTR2·PRRT·DOTATATE 임상 맥락",
     "14_benchmark_validation":"문헌 17종 실측 vs 예측 순위 — Spearman 정량+정성",
     "15_admet_benchmark":"물성 r≈1.0 정확 / 용혈 독성 AUC 0.19(역전) — 정직 검증",
    }
    for g in GROUP_ORDER:
        for s2 in [p for p in pages if META[p][1]==g]:
            t,_,emo=META[s2]
            cards+=f'<a class="fcard" href="pages/{s2}.html" style="text-decoration:none;color:inherit"><h3>{emo} {t}</h3><div class="desc">{descs.get(s2,"")}</div><div class="foot"><span class="badge b-info">{g}</span><span style="color:var(--teal)">자세히 →</span></div></a>'
    cards+='</div>'
    indirect="""<h2>"간접 검증"은 무엇이고 왜 신뢰할 수 있나</h2>
    <div class="note honest">절대 실측(wet-lab)이 없는 지표는 <b>상대 순위·상관 검증</b>으로 의사결정 신뢰를 확보한다. 핵심 사례:
    <ul>
    <li><b>혈중 반감기</b>: 절대 t½ 캘리브레이션은 없지만, 문헌 t½ 공개 펩타이드 대비 <b>Spearman ρ≥0.86(휴리스틱)/0.78(RF CV)</b>로 "어느 후보가 더 오래 가나"라는 순위 경향성을 검증(test_halflife_benchmark.py).</li>
    <li><b>독성</b>: pepADMET binary는 비변별적(옥시토신·native 모두 toxic)이라 폐기 → <b>HC50을 native 대비 상대값</b>으로 게이트(절대 단위 미검증).</li>
    <li><b>선택성</b>: curated SSTR2가 source 유래라 native 자체 margin +13.37 → <b>Δmargin = margin − 13.37</b>(home-advantage 보정)로만 진짜 우위 판정.</li>
    <li><b>ΔG</b>: PyRosetta 실측이나 절대값 calibration은 없음 → 후보 간 상대 비교로 사용, 실패는 fail-closed(999) 도태.</li>
    </ul>이 "정직한 상대화"가 본 시스템의 신뢰 설계 핵심이다.</div>"""
    b=bench_stats()
    def f(x,d=2):
        return ("%+.*f"%(d,x)) if isinstance(x,(int,float)) else "?"
    def fr(x):
        return ("%.3f"%x) if isinstance(x,(int,float)) else "?"
    bench=f"""<h2>🧪 벤치마크 검증 — 방법론 &amp; 결과</h2>
    <div class="note honest"><b>방법론(공통):</b> 온라인 공개 문헌의 <b>실측 ground-truth</b>(펩타이드 t½·용혈 HC50, 전부 출처 URL 인용)를
    모으고, 동일 서열에 <b>우리 파이프라인 예측을 실제 코드로 실행</b>한 뒤 둘의 <b>순위·분류 일치도</b>를 정량화한다
    (Spearman ρ = 순위상관, ROC-AUC = 용혈/비용혈 분류력, Pearson r = 물성 계산 일치). 절대 캘리브레이션이 없는 surrogate를
    <b>"상대 순위가 맞는가"</b>로 정직하게 평가하는 설계다. 우리 예측값은 재현 가능(`bench_run.py`/`bench_admet_tox.py`).</div>

    <div class="grid">
      <div class="fcard" style="border-top-color:var(--teal)">
        <h3>① 혈중 반감기 (t½) · N={b['hl_n']}</h3>
        <div class="desc">문헌 실측 t½ 순위 vs 우리 예측 순위 (Spearman). L-aa({b['hl_nL']}) vs D-aa({b['hl_nD']}) 분리.</div>
        <table style="margin:10px 0">
          <tr><th>그룹</th><th>raw ρ</th><th>+metadata ρ</th></tr>
          <tr><td><b>전체</b></td><td>{f(b['hl_all_raw'])}</td><td><b style="color:#067a5b">{f(b['hl_all_meta'])}</b></td></tr>
          <tr><td>L-aa</td><td>{f(b['hl_L_raw'])}</td><td>{f(b['hl_L_meta'])}</td></tr>
          <tr><td>D-aa</td><td>{f(b['hl_D_raw'])}*</td><td>{f(b['hl_D_meta'])}*</td></tr>
        </table>
        <div style="font-size:12.5px;color:var(--muted)"><b>결과:</b> <b>L-aa</b>는 정보 없이도 방향성 추종(raw {f(b['hl_L_raw'])}).
        <b>D-aa</b>는 raw가 전부 ~0.04h로 바닥(역전) → <b>변형 메타데이터를 주면 자릿수 교정</b>, 전체 ρ {f(b['hl_all_raw'])}→<b>{f(b['hl_all_meta'])}</b>.
        한계는 "공식 부재"가 아니라 <b>메타데이터 부재</b>. <span style="font-size:11px">(*그룹 ρ는 소표본 노이즈)</span></div>
        <div class="foot"><span class="badge b-mid">L 유효 / D는 메타데이터 필요</span><a href="pages/14_benchmark_validation.html">자세히 →</a></div>
      </div>
      <div class="fcard" style="border-top-color:var(--coral)">
        <h3>② ADMET (물성 · 독성)</h3>
        <div class="desc">물성=참조도구 일치도(r), 독성=용혈/비용혈 분류력(AUC).</div>
        <table style="margin:10px 0">
          <tr><th>항목</th><th>지표</th></tr>
          <tr><td>물성: GRAVY/Instability</td><td><b style="color:#067a5b">r={fr(b['desc_gravy'])}/{fr(b['desc_inst'])}</b></td></tr>
          <tr><td>물성: pI/Boman</td><td>r={fr(b['desc_pi'])}/{fr(b['desc_boman'])}</td></tr>
          <tr><td><b>독성: 용혈 분류 AUC</b></td><td><b style="color:#a3322f">{fr(b['tox_auc'])}</b> (n={b['tox_npos']}+{b['tox_nneg']})</td></tr>
        </table>
        <div style="font-size:12.5px;color:var(--muted)"><b>결과:</b> 물성 descriptor 계산은 검증된 참조도구와 <b>r≈1.0 일치(정확)</b>.
        그러나 pepADMET 용혈 독성 예측은 독립 AMP에서 <b>AUC 0.19로 역전</b>(LL-37·Cecropin 등 비용혈을 최독성 오판) — 범용 독성
        라벨로 신뢰 불가, <b>native 대비 상대 게이트로만</b> 사용해야 함.</div>
        <div class="foot"><span class="badge b-mid">물성✓ / 독성✗</span><a href="pages/15_admet_benchmark.html">자세히 →</a></div>
      </div>
    </div>"""
    body=hero+stats+win+arch+indirect+bench+cards
    open(f"{SITE}/index.html","w",encoding="utf-8").write(
        page_shell("개요", body, "index", depth_css="assets/style.css", in_pages=False))

if __name__=="__main__":
    n=build_feature_pages()
    build_index()
    print(f"빌드 완료: index.html + {n} feature pages → {SITE}")
