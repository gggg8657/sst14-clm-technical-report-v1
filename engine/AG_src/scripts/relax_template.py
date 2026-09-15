#!/usr/bin/env python3
"""relax_template.py
==================
boltz 예측 SSTR2-SST14 복합체처럼 strain 이 큰 시작 구조를 좌표 제약 하에서
FastRelax 하여 **안정적이고 재현성 있는 템플릿**을 만든다.

배경(2026-06-09): boltz 복합체를 직접 FlexPepDock 하면 pre_score~3365 의 큰 strain
때문에 시드마다 전혀 다른 local minimum 에 빠져 baseline ddG 분산이 ±150 REU 에 달함
→ 변이체 ΔG 신호가 노이즈에 묻힘. 한 번 relax 해 두면 후속 mutate+refine 이 일관된
바닥에서 출발해 분산이 줄어든다.

좌표 제약(CA, stdev 0.5A)을 걸어 전체 fold/binding pose 가 크게 흐트러지지 않게 하고,
이황화결합(Cys3-Cys14)은 relax 전 detect 하여 보존한다.

Usage:
    python relax_template.py --input complex.pdb --output relaxed.pdb [--cst-weight 1.0]

stdout: JSON {pre_score, post_score, disulfide_intact, sg_sg_distance}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def init_pyrosetta() -> None:
    import pyrosetta
    pyrosetta.init("-mute all -detect_disulf true", silent=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Coordinate-constrained FastRelax of a complex template")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cst-weight", type=float, default=1.0,
                        help="coordinate_constraint score weight (높을수록 원구조 고정)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(json.dumps({"error": f"input not found: {args.input}"}))
        sys.exit(1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    init_pyrosetta()
    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.scoring import ScoreType

    pose = pyrosetta.pose_from_pdb(args.input)

    # 이황화 자동 검출 + SG-SG 거리 (relax 전)
    def sg_sg_min(p):
        cys_sg = []
        for i in range(1, p.total_residue() + 1):
            r = p.residue(i)
            if r.name3() in ("CYS", "CYZ") and r.has("SG"):
                cys_sg.append(r.xyz("SG"))
        best = None
        for a in range(len(cys_sg)):
            for b in range(a + 1, len(cys_sg)):
                d = (cys_sg[a] - cys_sg[b]).norm()
                if best is None or d < best:
                    best = d
        return best

    pre_sg = sg_sg_min(pose)

    scorefxn = pyrosetta.get_fa_scorefxn()
    scorefxn.set_weight(ScoreType.coordinate_constraint, args.cst_weight)
    pre_score = scorefxn(pose)

    # FastRelax 가 constrain_relax_to_start_coords(True) 로 CA 좌표 제약을 내부 생성한다
    # (별도 AddConstraints mover 불필요 — API 호환성·중복 회피).
    relax = FastRelax(scorefxn, 5)  # 5 = standard repeats
    relax.constrain_relax_to_start_coords(True)  # CA 좌표 제약 자동 생성
    relax.apply(pose)

    post_score = scorefxn(pose)
    post_sg = sg_sg_min(pose)
    pose.dump_pdb(args.output)

    result = {
        "pre_score": round(pre_score, 3),
        "post_score": round(post_score, 3),
        "pre_sg_sg": round(pre_sg, 3) if pre_sg is not None else None,
        "sg_sg_distance": round(post_sg, 3) if post_sg is not None else None,
        "disulfide_intact": bool(post_sg is not None and post_sg < 2.5),
        "output": args.output,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
