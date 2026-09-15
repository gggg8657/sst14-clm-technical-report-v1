# =============================================================================
# electrostatics.pml
# 수용체 정전기 포텐셜 표면 위에 펩타이드 표시
# Electrostatic surface of receptor with peptide sticks overlay
#
# 주의 / Note:
#   정전기 계산은 APBS 또는 PyMOL의 내장 Coulomb 표면을 사용합니다.
#   Electrostatics computed via PyMOL's built-in Coulomb surface.
#   보다 정확한 계산을 위해서는 APBS 플러그인을 사용하세요.
#   For accurate calculations use the APBS plugin.
#
# 사용법 / Usage:
#   pymol -c electrostatics.pml
# 변수 / Variables (파이프라인에서 치환):
#   @RECEPTOR_PDB@  : 수용체 PDB 경로
#   @PEPTIDE_PDB@   : 펩타이드 PDB 경로
#   @OUTPUT_PNG@    : 출력 PNG 경로
# =============================================================================

reinitialize

# --- 구조 로드 ---
load @RECEPTOR_PDB@, receptor
load @PEPTIDE_PDB@, peptide

# --- 수소 추가 (정전기 계산에 필요) ---
# Add hydrogens for electrostatic calculation
h_add receptor

# --- 부분 전하 설정 (AMBER force field 기반) ---
# Assign partial charges using AMBER parameters
set_charges receptor

# =============================================================================
# 정전기 포텐셜 Ramp 정의
# Define color ramp for electrostatic potential:
#   음전하(빨강) <- 중성(흰색) -> 양전하(파랑)
#   Negative (red) <- Neutral (white) -> Positive (blue)
# =============================================================================
ramp_new elec_ramp, receptor, [-5, 0, 5], [red, white, blue]

# --- 수용체 표면 생성 및 정전기 색상 적용 ---
# Generate solvent-accessible surface and color by electrostatic potential
show surface, receptor
set surface_color, elec_ramp, receptor
set surface_transparency, 0.25, receptor    # 약간 투명 (펩타이드 가시성 확보)

# 수용체 카툰도 표시 (참조용)
show cartoon, receptor
set cartoon_transparency, 0.8, receptor
color gray60, receptor

# --- 펩타이드 바인더: sticks으로 표시 ---
hide everything, peptide
show sticks, peptide
color white, peptide                        # 흰색 배경과 대비
set stick_radius, 0.18, peptide
# 원소 색상으로 이종원자 구분 (N=파랑, O=빨강, S=노랑)
util.cbaw peptide

# --- 펩타이드 탄소만 초록으로 구분 ---
color limon, (peptide and name C*)

# --- 포켓 계면 잔기 반투명 sticks ---
select iface, byres (receptor within 4 of peptide)
show sticks, iface
set stick_transparency, 0.3, iface
util.cbac iface

# --- 수소결합 표시 (투명 표면 위에) ---
distance hbonds_elec, peptide, iface, 3.5, mode=2
color cyan, hbonds_elec
set dash_color, cyan, hbonds_elec
set dash_width, 2.0, hbonds_elec
hide labels, hbonds_elec

# --- 카메라 및 렌더링 설정 ---
orient all
zoom peptide, 10

set ray_shadows, 0                          # 표면 렌더링 시 그림자 제거
set ambient, 0.6
set specular, 0.6
set shininess, 40
set antialias, 2
bg_color black                              # 검은 배경 (정전기 색상 대비 강조)

ray 2400, 1800
png @OUTPUT_PNG@, dpi=300

quit
