# =============================================================================
# overview.pml
# SSTR2 수용체 + 펩타이드 바인더 전체 개요 렌더링
# Overview rendering: receptor (cartoon) + peptide binder (sticks)
#
# 사용법 / Usage:
#   pymol -c overview.pml -- receptor.pdb peptide.pdb output_dir/
# 또는 PyMOL 내부에서 / Or inside PyMOL:
#   run overview.pml
# =============================================================================

# --- 변수 설정 (파이프라인에서 치환됨) ---
# Variables to be substituted by the pipeline at runtime
# @RECEPTOR_PDB@  : 수용체 PDB 파일 경로
# @PEPTIDE_PDB@   : 펩타이드 PDB 파일 경로
# @OUTPUT_PNG@    : 출력 PNG 파일 경로 (기본: overview.png)

# --- 기존 세션 초기화 ---
reinitialize

# --- 수용체 로드 및 스타일 설정 ---
# Load receptor and set cartoon representation in light gray
load @RECEPTOR_PDB@, receptor
show cartoon, receptor
color gray80, receptor
set cartoon_transparency, 0.1, receptor

# --- 펩타이드 바인더 로드 및 스타일 설정 ---
# Load peptide binder and show as sticks
load @PEPTIDE_PDB@, peptide
show sticks, peptide
set stick_radius, 0.15, peptide

# --- 체인별 색상 지정 ---
# Color by chain (receptor = slate, peptide = hot pink)
util.cbc peptide
color slate, receptor

# --- pLDDT B-factor 스펙트럼 색상 (펩타이드) ---
# Spectrum coloring by B-factor (pLDDT) for peptide
# blue = low pLDDT (less confident), red = high pLDDT (confident)
spectrum b, blue_white_red, peptide, minimum=50, maximum=100

# --- 표면 설정 (수용체 반투명 표면) ---
show surface, receptor
set surface_transparency, 0.7, receptor
set surface_color, gray90, receptor

# --- 렌더링 설정 ---
# Rendering quality settings
set ray_shadows, 1
set ray_shadow_decay_factor, 0.1
set ambient, 0.4
set specular, 0.3
set antialias, 2
bg_color white

# --- 전체 구조 정렬 및 방향 설정 ---
orient all
zoom all, 5

# --- 광선 추적 렌더링 및 PNG 저장 ---
# Ray trace and save PNG (2400 x 1800 pixels)
ray 2400, 1800
png @OUTPUT_PNG@, dpi=300

quit
