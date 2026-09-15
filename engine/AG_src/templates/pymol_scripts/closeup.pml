# =============================================================================
# closeup.pml
# SSTR2 결합 포켓 근접 촬영 렌더링
# Closeup rendering of the binding pocket and peptide contacts
#
# 사용법 / Usage:
#   pymol -c closeup.pml
# 변수 / Variables (파이프라인에서 치환):
#   @RECEPTOR_PDB@      : 수용체 PDB 경로
#   @PEPTIDE_PDB@       : 펩타이드 PDB 경로
#   @POCKET_RESIDUES@   : 포켓 잔기 선택 문자열 (예: "resi 122+127+184+197+205+272+294")
#   @OUTPUT_PNG@        : 출력 PNG 경로
# =============================================================================

reinitialize

# --- 수용체 및 펩타이드 로드 ---
load @RECEPTOR_PDB@, receptor
load @PEPTIDE_PDB@, peptide

# --- 기본 표현: 수용체 cartoon, 펩타이드 sticks ---
hide everything
show cartoon, receptor
color gray70, receptor
show sticks, peptide
color hotpink, peptide
set stick_radius, 0.18, peptide

# --- 펩타이드 주변 5 Å 이내 포켓 잔기 선택 ---
# Select pocket residues within 5 Å of the peptide
select pocket_residues, (@POCKET_RESIDUES@) and receptor
select contact_shell, byres (receptor within 5 of peptide)

# 포켓 잔기를 sticks으로 표시 (탄소: 녹색, 나머지: 원소 색상)
show sticks, contact_shell
util.cbag contact_shell          # carbon = green
set stick_radius, 0.15, contact_shell

# 핫스팟 잔기 강조 (노란색)
select hotspots, pocket_residues and contact_shell
color yellow, hotspots
set stick_radius, 0.2, hotspots

# --- 접촉 잔기 레이블 ---
# Label contact residues with one-letter code + residue number
set label_size, 14
set label_color, black
label (contact_shell and name CA), "%s%s" % (oneletter, resi)

# --- 수소결합 표시 ---
# Show hydrogen bonds between peptide and pocket
distance hbonds, peptide, contact_shell, 3.5, mode=2
set dash_color, blue, hbonds
set dash_gap, 0.3, hbonds
set dash_width, 2.0, hbonds
hide labels, hbonds

# --- 카메라 설정: 포켓 중심으로 확대 ---
# Zoom to pocket center with buffer of 8 Å
zoom contact_shell, 8
orient contact_shell

# --- 렌더링 설정 ---
set ray_shadows, 1
set ambient, 0.5
set antialias, 2
bg_color white

ray 2400, 1800
png @OUTPUT_PNG@, dpi=300

quit
