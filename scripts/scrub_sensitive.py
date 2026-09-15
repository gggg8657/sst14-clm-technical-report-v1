#!/usr/bin/env python3
"""
scrub_sensitive.py — 보안 스크럽 스크립트
HTML/md/pdb 텍스트에서 민감 정보를 탐지하거나 치환합니다.

동작 모드:
  --dry-run (기본값): 치환 건수·파일별 매칭만 출력, 원본 미변경.
  --apply            : 실제 치환 수행 (주의: 원본 덮어씀).

치환 패턴:
  1. 로컬 절대경로  /home/<user>/...  → [LOCAL_PATH]
  2. KAERI 이메일   *@kaeri.re.kr     → [KAERI_EMAIL]
  3. PDB REMARK 라인 내 절대경로 제거  → (경로 부분만 삭제)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 패턴 정의
# ---------------------------------------------------------------------------

# 1. /home/<username>/... 로컬 절대경로  (그리디, 공백·따옴표·꺽쇠 앞에서 종료)
_LOCAL_PATH_PATTERN = re.compile(
    r"/home/[A-Za-z0-9_.-]+/[^\s\"'<>`,;)(\[\]]*"
)

# 2. *@kaeri.re.kr 이메일
_EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+\-]+@kaeri\.re\.kr"
)

# 3. PDB REMARK 라인 내 절대경로
#    행 시작이 "REMARK" 이고 그 안에 /home/... 포함 시 경로 부분만 제거
_PDB_REMARK_PATH_PATTERN = re.compile(
    r"(^REMARK\b.*?)(/home/[A-Za-z0-9_.-]+/[^\s\"'<>`,;)(\[\]]*)",
    re.MULTILINE,
)

# 치환 상수
_LOCAL_PATH_REPLACEMENT = "[LOCAL_PATH]"
_EMAIL_REPLACEMENT = "[KAERI_EMAIL]"

# 처리 대상 확장자
_TARGET_EXTENSIONS: frozenset[str] = frozenset(
    {".html", ".htm", ".md", ".markdown", ".pdb", ".py", ".txt", ".css", ".json", ".yaml", ".yml"}
)


# ---------------------------------------------------------------------------
# 핵심 함수
# ---------------------------------------------------------------------------


def scrub_text(text: str) -> tuple[str, dict[str, int]]:
    """
    텍스트에서 민감 정보를 치환하고 (결과 텍스트, 패턴별 치환 횟수) 반환.

    Returns:
        (scrubbed_text, counts)
        counts: {"local_path": int, "email": int, "pdb_remark": int}
    """
    counts: dict[str, int] = {"local_path": 0, "email": 0, "pdb_remark": 0}

    # 1) PDB REMARK 경로 (REMARK 라인 한정) — local_path보다 먼저 처리해 카운트 분리
    def _replace_pdb_remark(m: re.Match) -> str:
        counts["pdb_remark"] += 1
        return m.group(1)  # REMARK 접두사만 남기고 경로 제거

    text = _PDB_REMARK_PATH_PATTERN.sub(_replace_pdb_remark, text)

    # 2) 일반 로컬 절대경로
    def _replace_local_path(m: re.Match) -> str:
        counts["local_path"] += 1
        return _LOCAL_PATH_REPLACEMENT

    text = _LOCAL_PATH_PATTERN.sub(_replace_local_path, text)

    # 3) KAERI 이메일
    def _replace_email(m: re.Match) -> str:
        counts["email"] += 1
        return _EMAIL_REPLACEMENT

    text = _EMAIL_PATTERN.sub(_replace_email, text)

    return text, counts


def collect_files(root: Path) -> list[Path]:
    """
    루트 디렉토리에서 재귀적으로 대상 확장자 파일을 수집.
    바이너리 파일은 건너뜁니다.
    """
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _TARGET_EXTENSIONS:
            files.append(p)
    return files


def process_file(
    path: Path,
    dry_run: bool = True,
    encoding: str = "utf-8",
) -> Optional[dict[str, int]]:
    """
    단일 파일 처리.

    Args:
        path: 대상 파일 경로
        dry_run: True이면 원본 미변경 (카운트만 반환)
        encoding: 파일 인코딩

    Returns:
        {"local_path": int, "email": int, "pdb_remark": int}
        읽기 실패 시 None
    """
    try:
        original = path.read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        print(f"  [WARN] 읽기 실패: {path} — {exc}", file=sys.stderr)
        return None

    scrubbed, counts = scrub_text(original)
    total = sum(counts.values())

    if total == 0:
        return counts  # 매칭 없음 — 파일 쓰기 불필요

    if not dry_run:
        try:
            path.write_text(scrubbed, encoding=encoding)
        except OSError as exc:
            print(f"  [ERROR] 쓰기 실패: {path} — {exc}", file=sys.stderr)
            return None

    return counts


# ---------------------------------------------------------------------------
# 출력 포맷 헬퍼
# ---------------------------------------------------------------------------


def _fmt_counts(counts: dict[str, int]) -> str:
    return (
        f"local_path={counts['local_path']:3d} | "
        f"email={counts['email']:3d} | "
        f"pdb_remark={counts['pdb_remark']:3d}"
    )


def run(
    root: Path,
    dry_run: bool = True,
    verbose: bool = False,
) -> int:
    """
    메인 실행 루프.

    Returns:
        종료 코드 (0 = 정상)
    """
    mode_label = "DRY-RUN (원본 미변경)" if dry_run else "APPLY (원본 치환)"
    print(f"\n{'='*60}")
    print(f"scrub_sensitive.py  모드: {mode_label}")
    print(f"대상 루트: {root}")
    print(f"{'='*60}\n")

    files = collect_files(root)
    if not files:
        print("[INFO] 대상 파일 없음.")
        return 0

    total_counts: dict[str, int] = {"local_path": 0, "email": 0, "pdb_remark": 0}
    matched_files: list[tuple[Path, dict[str, int]]] = []

    for path in files:
        counts = process_file(path, dry_run=dry_run)
        if counts is None:
            continue
        total = sum(counts.values())
        if verbose or total > 0:
            tag = "[HIT]" if total > 0 else "    "
            print(f"  {tag} {_fmt_counts(counts)}  {path.relative_to(root)}")
        if total > 0:
            matched_files.append((path, counts))
            for k in total_counts:
                total_counts[k] += counts[k]

    print(f"\n{'─'*60}")
    print(f"  총 검사 파일   : {len(files):4d}")
    print(f"  매칭 파일      : {len(matched_files):4d}")
    print(f"  로컬경로 치환  : {total_counts['local_path']:4d} 건")
    print(f"  이메일 치환    : {total_counts['email']:4d} 건")
    print(f"  PDB REMARK 제거: {total_counts['pdb_remark']:4d} 건")
    print(f"  합계           : {sum(total_counts.values()):4d} 건")
    print(f"{'─'*60}")
    if dry_run:
        print("  [DRY-RUN] 원본 파일 무변경 완료.")
    else:
        print(f"  [APPLY] {len(matched_files)}개 파일 치환 완료.")
    print()

    return 0


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "HTML/md/pdb 텍스트에서 로컬 절대경로·KAERI 이메일·PDB REMARK 경로를 스크럽합니다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # dry-run (기본): 매칭 건수만 출력, 원본 미변경
  python scrub_sensitive.py /path/to/docs/

  # apply: 실제 치환 (주의: 원본 덮어씀)
  python scrub_sensitive.py /path/to/docs/ --apply

  # 매칭 없는 파일도 모두 출력
  python scrub_sensitive.py /path/to/docs/ --verbose
""",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="스크럽 대상 루트 디렉토리 경로",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="(기본값) 매칭 건수만 출력, 원본 미변경",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="실제 치환 수행 — 원본 파일 덮어씀 (주의)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="매칭 없는 파일도 모두 출력",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"[ERROR] 경로가 존재하지 않습니다: {root}", file=sys.stderr)
        sys.exit(1)
    if not root.is_dir():
        print(f"[ERROR] 디렉토리가 아닙니다: {root}", file=sys.stderr)
        sys.exit(1)

    # --apply가 명시된 경우에만 dry_run=False
    dry_run = not args.apply

    sys.exit(run(root=root, dry_run=dry_run, verbose=args.verbose))


if __name__ == "__main__":
    main()
