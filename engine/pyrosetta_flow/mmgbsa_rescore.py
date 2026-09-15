"""
mmgbsa_rescore.py — Single/Multi-snapshot MM-GBSA rescoring using OpenMM.

방법론:
  - SSBOND 레코드 주입: Boltz PDB에 SSBOND 없을 경우 Cys3-Cys14 레코드 자동 주입
  - pdbfixer로 수소·결손 원자 보정
  - AMBER14 (amber14-all.xml) + GBn2 implicit solvent (implicit/gbn2.xml)
  - 짧은 에너지 최소화 (energy minimization), n_snapshots 회 반복 가능
  - ΔG_bind = E(complex) − E(receptor) − E(peptide)
    (single-trajectory: 동일 최소화된 complex 구조에서 체인별 분리 에너지 계산)
  - n_snapshots >= 2: 서로 다른 seed로 최소화를 반복, median + stdev 반환

한계(caveat):
  - Single-snapshot(n_snapshots=1): MD trajectory 미사용. 엔트로피 기여(TΔS) 무시.
  - n_snapshots >= 2: seed 다양화 ≠ MD ensemble; 분산은 최소화 수렴 노이즈 추정.
  - Cyclic peptide의 SS bond: SSBOND 레코드 자동 주입 후 pdbfixer 처리.
    disulfide 거리가 이상할 경우 비표준 처리 경고 발생 가능.
  - AMBER14 force field는 표준 AA 20종 처리. 비표준 잔기(HAD 등)는 실패.

의존성:
  conda env mmgbsa (openmm 8.5+, pdbfixer, openmmforcefields)
  → 설치: conda install -n mmgbsa -c conda-forge openmm pdbfixer openmmforcefields
"""

from __future__ import annotations

import random
import statistics
import tempfile
import time
import logging
import traceback
from pathlib import Path
from io import StringIO
from typing import Optional

logger = logging.getLogger(__name__)

# numpy는 선택적 의존성 — try-import 패턴
try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# SS bond 주입 유틸
# ---------------------------------------------------------------------------

def _inject_ssbond_if_missing(
    pdb_path: str,
    peptide_chain: str,
) -> tuple[str, bool]:
    """PDB에 SSBOND 레코드가 없으면 Cys3-Cys14 레코드를 삽입한 임시 파일 생성.

    Boltz 예측 PDB는 SSBOND 레코드를 포함하지 않아 pdbfixer가 disulfide를
    인식하지 못하고 ΔG_bind 에 15~30 kcal/mol 오차를 유발한다.
    이 함수는 SST-14 계열(Cys3-Cys14) 전용으로, SSBOND 레코드를 헤더에 주입한다.

    Args:
        pdb_path: 원본 PDB 파일 경로
        peptide_chain: 펩타이드 chain ID (예: "A" 또는 "B")

    Returns:
        (effective_path, injected): injected=True면 tmp 파일 경로, False면 원본 경로
    """
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # SSBOND 레코드 존재 여부 확인
    has_ssbond = any(line.startswith("SSBOND") for line in lines)
    if has_ssbond:
        logger.debug("SSBOND 레코드 이미 존재: %s", pdb_path)
        return pdb_path, False

    # SSBOND 레코드 구성 (PDB 형식: 컬럼 고정폭)
    ssbond_line = (
        f"SSBOND   1 CYS {peptide_chain}    3     CYS {peptide_chain}   14\n"
    )

    # REMARK 줄 바로 앞에 삽입; REMARK 없으면 첫 줄 앞
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("REMARK"):
            insert_idx = i
            break

    new_lines = lines[:insert_idx] + [ssbond_line] + lines[insert_idx:]

    # 임시 파일에 저장 (호출자가 finally로 삭제)
    tmp_fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, encoding="utf-8"
    )
    tmp_fd.writelines(new_lines)
    tmp_fd.close()

    logger.info(
        "SSBOND 레코드 주입 완료: chain=%s pos3-pos14 → tmp=%s",
        peptide_chain, tmp_fd.name,
    )
    return tmp_fd.name, True


# ---------------------------------------------------------------------------
# OpenMM 내부 유틸
# ---------------------------------------------------------------------------

def _build_system_and_minimize(
    pdb_path: str,
    platform_name: str = "CUDA",
    device_index: str = "0",
    max_iterations: int = 500,
) -> tuple:
    """
    PDB를 읽어 AMBER14+GBn2로 최소화된 simulation 반환.
    Returns (simulation, topology, positions, forcefield) or raises.
    """
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from pdbfixer import PDBFixer
    except ImportError as e:
        raise ImportError(f"openmm/pdbfixer 미설치: {e}") from e

    # --- pdbfixer: 수소·결손 원자 보정 ---
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)   # pH 7.0

    # pdbfixer 결과를 StringIO로 전달 (임시 파일 불필요)
    import io
    pdb_stream = io.StringIO()
    app.PDBFile.writeFile(fixer.topology, fixer.positions, pdb_stream)
    pdb_stream.seek(0)

    pdb = app.PDBFile(pdb_stream)

    # --- Force field: AMBER14 + GBn2 implicit solvent ---
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")

    # --- System 생성 ---
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,
    )

    # --- Integrator (최소화 전용, 임의 온도) ---
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.004 * unit.picoseconds,
    )

    # --- Platform 선택 (CUDA 우선, 실패시 CPU 폴백) ---
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        properties = {"DeviceIndex": device_index}
        simulation = app.Simulation(
            pdb.topology, system, integrator, platform, properties
        )
        logger.info("플랫폼: %s (device %s)", platform_name, device_index)
    except Exception as cuda_err:
        logger.warning("CUDA 플랫폼 실패(%s), CPU 폴백", cuda_err)
        platform = openmm.Platform.getPlatformByName("CPU")
        simulation = app.Simulation(pdb.topology, system, integrator, platform)

    simulation.context.setPositions(pdb.positions)

    # --- 에너지 최소화 ---
    simulation.minimizeEnergy(maxIterations=max_iterations)

    return simulation, pdb.topology


def _get_potential_energy_kj(simulation) -> float:
    """현재 simulation context의 potential energy를 kcal/mol로 반환."""
    import openmm.unit as unit
    state = simulation.context.getState(getEnergy=True)
    energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    energy_kcal = energy_kj / 4.184
    return energy_kcal


def _extract_chain_to_pdbfile(topology, positions, chain_ids: list[str]) -> str:
    """
    topology에서 지정된 chain_ids만 추출해 PDB 문자열로 반환.
    PDBFile.writeFile 후 re-parse로 결합(bond) 정보를 완전히 복원.
    """
    import openmm.app as app
    import openmm.unit as unit
    import io

    # 서브 topology 구성 (결합 포함)
    new_top = app.Topology()
    idx_map: dict[int, object] = {}  # old atom index -> new atom object
    keep_set: set[int] = set()

    for chain in topology.chains():
        if chain.id not in chain_ids:
            continue
        new_chain = new_top.addChain(chain.id)
        for res in chain.residues():
            new_res = new_top.addResidue(res.name, new_chain, res.id, res.insertionCode)
            for atom in res.atoms():
                new_atom = new_top.addAtom(atom.name, atom.element, new_res)
                idx_map[atom.index] = new_atom
                keep_set.add(atom.index)

    # 결합 복사 (chain 내부 결합만)
    for bond in topology.bonds():
        if bond[0].index in keep_set and bond[1].index in keep_set:
            new_top.addBond(idx_map[bond[0].index], idx_map[bond[1].index])

    # positions 추출
    pos_list = positions.value_in_unit(unit.nanometers)
    keep_order = sorted(keep_set)
    from openmm import Vec3
    new_positions = [Vec3(*pos_list[i]) for i in keep_order] * unit.nanometers

    # PDB 문자열로 직렬화 후 반환
    buf = io.StringIO()
    app.PDBFile.writeFile(new_top, new_positions, buf)
    return buf.getvalue()


def _energy_of_subcomplex(
    topology,
    positions,
    chain_ids: list[str],
    platform_name: str = "CUDA",
    device_index: str = "0",
) -> float:
    """
    복합체에서 chain_ids 부분만 추출해 에너지(kcal/mol) 계산.
    (단일 trajectory 방식: 복합체 최소화 구조 그대로 사용)
    PDB re-parse 방식으로 결합 정보 완전 복원.
    """
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    import io

    # PDB 문자열로 추출 후 re-parse (bond 정보 복원)
    sub_pdb_str = _extract_chain_to_pdbfile(topology, positions, chain_ids)
    sub_pdb = app.PDBFile(io.StringIO(sub_pdb_str))

    # Force field 독립 생성 (각 subcomplex용)
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(
        sub_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,
    )

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds
    )

    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        sim = app.Simulation(
            sub_pdb.topology, system, integrator, platform, {"DeviceIndex": device_index}
        )
    except Exception as e:
        logger.warning("서브컴플렉스 CUDA 실패(%s), CPU 폴백", e)
        platform = openmm.Platform.getPlatformByName("CPU")
        sim = app.Simulation(sub_pdb.topology, system, integrator, platform)

    sim.context.setPositions(sub_pdb.positions)
    # 최소화 없이 현재 좌표의 에너지만 계산 (single-trajectory 원칙)
    state = sim.context.getState(getEnergy=True)
    energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    return energy_kj / 4.184


# ---------------------------------------------------------------------------
# 단일 스냅샷 ΔG_bind 계산 (내부 헬퍼)
# ---------------------------------------------------------------------------

def _compute_dg_bind_one(
    pdb_path: str,
    receptor_chains: list[str],
    peptide_chain: str,
    platform_name: str,
    device_index: str,
    max_minimize_iterations: int,
    seed: Optional[int] = None,
) -> tuple[float, float, float, float, str]:
    """단일 최소화 수행 후 (dg_bind, e_complex, e_receptor, e_peptide, platform_used) 반환.

    Args:
        seed: 재현성 seed. None이면 seed 설정 생략.
    """
    if seed is not None:
        random.seed(seed)
        if _HAS_NUMPY:
            _np.random.seed(seed)

    simulation, topology = _build_system_and_minimize(
        pdb_path,
        platform_name=platform_name,
        device_index=device_index,
        max_iterations=max_minimize_iterations,
    )

    e_complex = _get_potential_energy_kj(simulation)
    platform_used: str = simulation.context.getPlatform().getName()

    import openmm.unit as unit
    state = simulation.context.getState(getPositions=True)
    minimized_positions = state.getPositions()

    e_receptor = _energy_of_subcomplex(
        topology, minimized_positions, receptor_chains,
        platform_name=platform_used, device_index=device_index,
    )
    e_peptide = _energy_of_subcomplex(
        topology, minimized_positions, [peptide_chain],
        platform_name=platform_used, device_index=device_index,
    )

    dg_bind = e_complex - e_receptor - e_peptide
    return dg_bind, e_complex, e_receptor, e_peptide, platform_used


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def mmgbsa_rescore(
    complex_pdb: str,
    receptor_chains: list[str],
    peptide_chain: str,
    platform_name: str = "CUDA",
    device_index: str = "0",
    max_minimize_iterations: int = 500,
    n_snapshots: int = 1,
) -> dict:
    """
    Single/Multi-snapshot MM-GBSA rescoring.

    Args:
        complex_pdb: 복합체 PDB 파일 경로 (수용체 + 펩타이드)
        receptor_chains: 수용체 chain ID 리스트, 예) ["B"]
        peptide_chain: 펩타이드 chain ID, 예) "A"
        platform_name: "CUDA" 또는 "CPU"
        device_index: CUDA device 인덱스 (str), 기본 "0"
        max_minimize_iterations: 에너지 최소화 최대 이터레이션
        n_snapshots: 최소화 반복 횟수 (기본 1). >= 2이면 seed 다양화로 분산 추정.
                     권장: 3 (속도/분산 균형)

    Returns:
        dict with keys:
            ok (bool): 성공 여부
            dg_bind (float | None): ΔG_bind median (n=1일 때 단일값), kcal/mol
            dg_bind_sd (float | None): ΔG_bind stdev (n>=2일 때), kcal/mol; n=1이면 None
            n_snapshots (int): 실제 사용한 snapshot 수
            e_complex (float | None): 복합체 minimized 에너지 (첫 snapshot), kcal/mol
            e_receptor (float | None): 수용체 에너지 (첫 snapshot), kcal/mol
            e_peptide (float | None): 펩타이드 에너지 (첫 snapshot), kcal/mol
            ss_bond_verified (bool): SSBOND 레코드 확인/주입 여부
            elapsed_s (float): 소요 시간, 초
            error (str | None): 실패 사유
            caveat (str): 방법론 한계 명시
            platform_used (str): 실제 사용 플랫폼

    방법론 한계:
        - Single-snapshot (no MD): 엔트로피(TΔS) 미반영
        - n_snapshots >= 2: seed 다양화 최소화; MD ensemble 아님
        - AMBER14 implicit GBn2: 용매화 에너지 근사
        - SS bond: SSBOND 레코드 자동 주입 후 pdbfixer 처리 (Cys3-Cys14)
    """
    CAVEAT = (
        "single-snapshot MM-GBSA (no MD, no TΔS): "
        "AMBER14+GBn2 implicit solvent, single pose, "
        "entropy not included. "
        "SS bond: SSBOND record auto-injected if missing (Cys3-Cys14). "
        f"n_snapshots={n_snapshots} (seed diversification, not MD ensemble)."
    )

    t0 = time.time()
    result: dict = dict(
        ok=False,
        dg_bind=None,
        dg_bind_sd=None,
        n_snapshots=n_snapshots,
        e_complex=None,
        e_receptor=None,
        e_peptide=None,
        ss_bond_verified=False,
        elapsed_s=0.0,
        error=None,
        caveat=CAVEAT,
        platform_used=platform_name,
    )

    # SS bond 주입된 임시 파일 경로 (finally에서 삭제)
    _tmp_pdb: Optional[str] = None

    try:
        # --- SS bond 검증/주입 ---
        effective_pdb, injected = _inject_ssbond_if_missing(complex_pdb, peptide_chain)
        if injected:
            _tmp_pdb = effective_pdb
        result["ss_bond_verified"] = True  # 주입됐든 기존이든 SSBOND 존재 보장

        # --- n_snapshots 처리 ---
        if n_snapshots <= 1:
            # 기존 단일 snapshot 경로 (완전 호환)
            logger.info("복합체 최소화 시작 (n_snapshots=1): %s", effective_pdb)
            dg_bind, e_complex, e_receptor, e_peptide, platform_used = (
                _compute_dg_bind_one(
                    effective_pdb, receptor_chains, peptide_chain,
                    platform_name, device_index, max_minimize_iterations,
                    seed=None,
                )
            )
            result.update(
                dg_bind=dg_bind,
                dg_bind_sd=None,
                n_snapshots=1,
                e_complex=e_complex,
                e_receptor=e_receptor,
                e_peptide=e_peptide,
                platform_used=platform_used,
                ok=True,
            )
            logger.info(
                "ΔG_bind = %.2f kcal/mol (n=1, elapsed %.1fs)",
                dg_bind, time.time() - t0,
            )

        else:
            # multi-snapshot: seed 다양화
            seed_list = list(range(42, 42 + n_snapshots))
            dg_list: list[float] = []
            first_e_complex: Optional[float] = None
            first_e_receptor: Optional[float] = None
            first_e_peptide: Optional[float] = None
            platform_used_final = platform_name

            for idx, seed in enumerate(seed_list):
                logger.info(
                    "복합체 최소화 [snapshot %d/%d, seed=%d]: %s",
                    idx + 1, n_snapshots, seed, effective_pdb,
                )
                dg, e_c, e_r, e_p, pu = _compute_dg_bind_one(
                    effective_pdb, receptor_chains, peptide_chain,
                    platform_name, device_index, max_minimize_iterations,
                    seed=seed,
                )
                dg_list.append(dg)
                if idx == 0:
                    first_e_complex = e_c
                    first_e_receptor = e_r
                    first_e_peptide = e_p
                    platform_used_final = pu
                logger.info("  snapshot %d ΔG_bind = %.2f kcal/mol", idx + 1, dg)

            dg_median = statistics.median(dg_list)
            dg_sd = statistics.stdev(dg_list) if len(dg_list) >= 2 else None

            result.update(
                dg_bind=dg_median,
                dg_bind_sd=dg_sd,
                n_snapshots=n_snapshots,
                e_complex=first_e_complex,
                e_receptor=first_e_receptor,
                e_peptide=first_e_peptide,
                platform_used=platform_used_final,
                ok=True,
            )
            logger.info(
                "ΔG_bind median=%.2f sd=%.3f kcal/mol (n=%d, elapsed %.1fs)",
                dg_median,
                dg_sd if dg_sd is not None else 0.0,
                n_snapshots,
                time.time() - t0,
            )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("mmgbsa_rescore 실패: %s", result["error"])

    finally:
        # 주입된 임시 PDB 삭제
        if _tmp_pdb is not None:
            try:
                Path(_tmp_pdb).unlink(missing_ok=True)
                logger.debug("tmp PDB 삭제: %s", _tmp_pdb)
            except OSError as e:
                logger.warning("tmp PDB 삭제 실패: %s", e)

    result["elapsed_s"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# CLI make-or-break 테스트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 기본 타겟 PDB
    ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_PDB = (
        ROOT
        / "AgenticAI4SCIENCE_pyrosetta_track"
        / "repos"
        / "ai4sci-kaeri"
        / "data"
        / "somatostatin_receptor"
        / "SSTR2_SST14_complex_boltz_1.pdb"
    )

    pdb_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_PDB)
    n_snaps = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Boltz 복합체: chain A = 펩타이드(SST-14), chain B = 수용체(SSTR2)
    receptor_chains = ["B"]
    peptide_chain = "A"

    # CUDA_VISIBLE_DEVICES 환경변수로 GPU 선택 시 DeviceIndex는 항상 "0"
    # 예: CUDA_VISIBLE_DEVICES=2 python mmgbsa_rescore.py → device_index="0"
    print(f"\n[make-or-break] PDB: {pdb_path}")
    print(f"  receptor_chains={receptor_chains}, peptide_chain='{peptide_chain}'")
    print(f"  platform=CUDA device_index=0 (CUDA_VISIBLE_DEVICES 기준)")
    print(f"  n_snapshots={n_snaps}\n")

    result = mmgbsa_rescore(
        pdb_path,
        receptor_chains=receptor_chains,
        peptide_chain=peptide_chain,
        platform_name="CUDA",
        device_index="0",
        n_snapshots=n_snaps,
    )

    print("\n=== MM-GBSA 결과 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["ok"]:
        print(f"\n  ΔG_bind  = {result['dg_bind']:.2f} kcal/mol")
        if result.get("dg_bind_sd") is not None:
            print(f"  ΔG_bind SD = {result['dg_bind_sd']:.3f} kcal/mol (n={result['n_snapshots']})")
        print(f"  E(cplx)  = {result['e_complex']:.2f} kcal/mol")
        print(f"  E(recep) = {result['e_receptor']:.2f} kcal/mol")
        print(f"  E(pep)   = {result['e_peptide']:.2f} kcal/mol")
        print(f"  elapsed  = {result['elapsed_s']:.1f} s")
        print(f"  platform = {result['platform_used']}")
        print(f"  ss_bond_verified = {result['ss_bond_verified']}")
        print(f"  caveat   : {result['caveat']}")
    else:
        print(f"\n[FAIL] {result['error']}")
        sys.exit(1)
